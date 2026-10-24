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
"""Timing studies in rounds, for decisions that hold up to the noise: the re-fit sweep (REFIT, sweep.py's preset) and
the held-out test (617 real-model shapes, after a rule is fitted).

    source v27_streamk_tile/env.sh
    python v27_streamk_tile/refit.py init --gpus 0,1,2,3            # manifest: hashes, blocks, deal, estimate
    bash v27_streamk_tile/run_refit.sh                               # warm, rounds 1-2, close, round 3, merge
    python v27_streamk_tile/refit.py shard --round 1 --slot 0        # one GPU's share of a round (run_refit.sh)
    python v27_streamk_tile/refit.py close                           # round 3's points, from rounds 1-2
    python v27_streamk_tile/refit.py merge                           # results/refit.csv, results/refit_report.md
    python v27_streamk_tile/refit.py init --study heldout --rule my_rule.py --gpus 0,1,2,3

init writes the run manifest (results/<prefix>_manifest.json): the sha256 of sweep.FROZEN_FILES (every timing chunk
re-hashes them and stops on a change), the Triton and repository commits, the GPUs, the start time, every shape's block
of points and its GPU slot. Comparisons are only ever made within one block, one GPU and one round:
- **Blocks:** every point of a shape (all its tiles and configurations) is timed on one GPU in a round, in a shuffled
  order (seeded per round and shape); decisions use ratios within the block, so a GPU's offset cancels.
- **Deal:** whole shapes go to GPU slots by estimated cost (sweep.point_cost, longest first onto the least loaded
  slot), so the shards finish together. In round r, shape s runs on slot (g(s) + r - 1) % G, so it sees a different
  GPU in each round. The order of the shapes is reshuffled per round and slot, so a point's repeats are never back to
  back.
- **Rounds 1 and 2:** every point. rel_r(p) = time / the geometric mean of the shape's normalising set in round r (all
  its points; for a shape in round 3, its round-3 points), so rounds compare without GPU offsets. The noise threshold
  sigma is the 90th percentile of |rel_1 / rel_2 - 1|.
- **Round 3 (close):** a third timing, on a GPU slot neither round used, of every point that could change a decision:
  a tile's best and runner-up configurations within 2 sigma, a shape's best and runner-up tiles (their best
  configurations) within 2 sigma, and any point whose two rounds differ by more than sigma; with the shape's best
  points of its 3 best tiles as anchors of the block.
- **Merge:** rel(p) = the median over its rounds; cfg_ratio = rel / the tile's best explicit (non-cap)
  configuration, tile_ratio = the tile's best / the shape's best; the report gives the noise, per-GPU offsets, the
  position-in-chunk effect, round 3's share and what it changed, how close the winners are, and the coverage per tile
  class in the tiles' own coordinates.

Studies (--study):
- refit: sweep.refit_points() (the REFIT tiles that passed their check on this kernel). A shard leaves out the
  points whose warm compile failed (results/<prefix>_warm_fail.json: a pdp{P} build that does not fit, say).
- refit2: sweep.refit2_points(), the shapes REFIT did not cover (tall-skinny, large and wide-short compute-bound, a
  denser memory-bound grid to M = 448, and structured shapes at exact tile counts around half a wave on auto_tile's
  path), seeded with sweep.REFIT2_SEED; the manifest records each shape's group, each structured shape's target tile
  and the test split (sweep.refit2_test(), held back from fitting).
- heldout: the 617 distinct shapes of sweep.AITER_SHAPES_CSV, never in REFIT's grid. Per shape: the rule's pick
  (--rule FILE with pick(M, N, K) -> (kind, tile); left out if not given), v26 as deployed (cap at 256x256x64), and a
  reduced factorial: the REFIT quadrant tiles x their dp, pdp, pdp{P} (where checked), cap and HELDOUT_TOP most
  often near-best configurations on REFIT's compute-bound shapes (results/<refit-prefix>.csv), for the regret. None of
  these shapes is memory-bound (M >= 512), so they cannot test the memory-bound rule.
- worst10: sweep.worst10_points(), the AITER shapes where v27's auto tile is 10% or more behind the best library, plus
  control shapes where it is ahead (sweep.WORST10_CSV): every quadrant tile with every valid configuration except the
  atomic ending, including bf16 and bf16-pair partials. The manifest records each shape's group and auto_tile.
"""

import argparse
import collections
import csv
import glob
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep  # noqa: E402

RES = os.path.join(HERE, "results")
HELDOUT_TOP = 4
CLOSE_FACTOR = 2


def shape_key(sh):
    return "x".join(map(str, sh))


def parse_shape(k):
    return tuple(int(x) for x in k.split("x"))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))] if xs else float("nan")


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def manifest_path(prefix):
    return os.path.join(RES, f"{prefix}_manifest.json")


def load_manifest(prefix):
    return json.load(open(manifest_path(prefix)))


def git(*args, cwd=sweep.A16):
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def triton_info():
    import triton

    root = os.path.dirname(os.path.dirname(os.path.dirname(triton.__file__)))
    return {"version": triton.__version__, "path": triton.__file__, "commit": git("rev-parse", "HEAD", cwd=root),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no", cwd=root))}


def tile_class(tile):
    geo = sweep.v27_module().tile_geometry(tile)
    return "quadrant" if not geo["small"] else f"small P={geo['num_programs']}"


def regime(M):
    return "compute-bound" if M >= 512 else "memory-bound"


# ---------------------------------------------------------------------------------------------------- studies


def refit_blocks(args):
    return {shape_key(sh): pts for sh, pts in sweep.refit_points(checked=not args.unchecked).items()}, {}


def refit2_blocks(args):
    """REFIT2: sweep.refit2_points(), with each shape's group and the test split (held back from fitting) recorded."""
    groups = {shape_key(sh): g for g, v in sweep.refit2_shapes().items() for sh in v}
    blocks = {shape_key(sh): pts for sh, pts in sweep.refit2_points(checked=not args.unchecked).items()}
    return blocks, {"groups": groups, "test": sorted(shape_key(sh) for sh in sweep.refit2_test()),
                    "structured_targets": {shape_key(sh): f"{bm}x{bn}" for sh, (bm, bn) in
                                           sorted(sweep.refit2_structured().items())},
                    "mb_prune": sweep.REFIT2_MB_PRUNE,
                    "unchecked_raggedm": [t for t in sweep.REFIT_TILES if sweep.shape_info(t, 1024, 1024, 4096)["small"]
                                          and not sweep.raggedm_checked(t)]}


def warm_failures(prefix):
    """{point: error} of the points whose warm compile failed (results/<prefix>_warm_fail.json, run_refit.sh)."""
    path = os.path.join(RES, f"{prefix}_warm_fail.json")
    return json.load(open(path))["failed"] if os.path.exists(path) else {}


def near_best_configs(prefix, top):
    """Per quadrant tile, its `top` explicit split configurations most often within the noise of the tile's best on
    REFIT's compute-bound shapes (merged results/<prefix>.csv, the noise from <prefix>_round3.json)."""
    rows = [r for r in csv.DictReader(open(os.path.join(RES, f"{prefix}.csv")))
            if regime(int(r["M"])) == "compute-bound" and r["cfg_ratio"]]
    noise = json.load(open(round3_path(prefix)))["noise"]
    cnt = collections.defaultdict(collections.Counter)
    for r in rows:
        if (r["kind"] not in ("dp", "pdp", "cap") and not sweep.pdp_programs(r["kind"])
                and float(r["cfg_ratio"]) <= 1 + noise):
            cnt[r["tile"]][r["kind"]] += 1
    return {t: [k for k, _ in c.most_common(top)] for t, c in cnt.items()}


def heldout_blocks(args):
    import importlib.util

    rule = None
    if args.rule:
        spec = importlib.util.spec_from_file_location("heldout_rule", args.rule)
        rule = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rule)
    top = near_best_configs(args.refit_prefix, HELDOUT_TOP)
    quad = [t for t in sweep.REFIT_TILES if not sweep.shape_info(t, 1024, 1024, 4096)["small"]
            and sweep.refit_checked(t)]
    blocks = {}
    for sh in sweep.heldout_shapes():
        pts = [sweep.point_name("cap", "256x256x64", *sh)]
        if rule:
            kind, tile = rule.pick(*sh)
            pts.append(sweep.point_name(kind, tile, *sh))
        for t in quad:
            pdp = [k for k in ("pdp512", "pdp768", "pdp1024") if sweep.refit_checked(t, sweep.pdp_programs(k))]
            for k in ["dp", "pdp", "cap"] + pdp + top.get(t, []):
                if sweep.valid(k, t, *sh):
                    pts.append(sweep.point_name(k, t, *sh))
        blocks[shape_key(sh)] = sorted(set(pts))
    picks = {shape_key(sh): rule.pick(*sh) for sh in sweep.heldout_shapes()} if rule else {}
    sha = (lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None)
    return blocks, {"rule": os.path.abspath(args.rule) if args.rule else None, "top_configs": top,
                    "refit_prefix": args.refit_prefix, "picks": picks,
                    "rule_sha256": sha(args.rule) if args.rule else None,
                    "rule_table_sha256": sha(getattr(rule, "TABLE", "")) if rule else None}


def worst10_blocks(args):
    """WORST10: sweep.worst10_points(), one block per shape; each shape's group (worst or control) and the points the
    checks left out are recorded. The deployed v27 (cap at auto_tile) and v26 (cap at 256x256x64) are in every block."""
    blocks = {shape_key(sh): pts for sh, pts in sweep.worst10_points(checked=not args.unchecked).items()}
    allpts = {shape_key(sh): pts for sh, pts in sweep.worst10_points(checked=False).items()}
    v27 = sweep.v27_module()
    return blocks, {"groups": {shape_key(sh): g for sh, g in sweep.worst10_shapes().items()},
                    "shapes_csv": os.path.abspath(sweep.WORST10_CSV),
                    "shapes_csv_sha256": hashlib.sha256(open(sweep.WORST10_CSV, "rb").read()).hexdigest(),
                    "auto_tile": {shape_key(sh): v27.auto_tile(*sh) for sh in sweep.worst10_shapes()},
                    "unchecked_points": sum(len(allpts[k]) - len(blocks[k]) for k in blocks)}


STUDIES = {"refit": refit_blocks, "refit2": refit2_blocks, "heldout": heldout_blocks, "worst10": worst10_blocks}
SEEDS = {"refit2": sweep.REFIT2_SEED}


# ---------------------------------------------------------------------------------------------------- init


def deal(blocks, slots):
    """{shape: slot}: shapes by estimated cost, longest first, each onto the least loaded slot."""
    cost = {k: sum(sweep.point_cost(p) for p in pts) for k, pts in blocks.items()}
    load = [0.0] * slots
    out = {}
    for k in sorted(cost, key=lambda k: (-cost[k], k)):
        i = min(range(slots), key=lambda i: (load[i], i))
        out[k] = i
        load[i] += cost[k]
    return out, load


def preflight(gpus):
    """Other users of the machine: stress checks or sweeps (pgrep), and processes on the chosen GPUs (rocm-smi)."""
    me = str(os.getpid())
    p = subprocess.run(["pgrep", "-af", "check_grid|check_streamk|check_queue|sweep.py|refit.py"],
                       capture_output=True, text=True)
    others = [ln for ln in p.stdout.splitlines() if ln.split()[0] != me and "pgrep" not in ln]
    foreign = {g: sweep.foreign_pids(g) for g in gpus}
    return others, {g: v for g, v in foreign.items() if v}


def init(args):
    prefix = args.prefix or args.study
    path = manifest_path(prefix)
    if os.path.exists(path) and not args.force:
        sys.exit(f"{path} exists (a run in progress?); --force to replace it")
    gpus = args.gpus.split(",")
    others, foreign = preflight(gpus)
    for ln in others[:10]:
        print(f"PREFLIGHT: another check or sweep is running: {ln[:160]}")
    if len(others) > 10:
        print(f"PREFLIGHT: ... and {len(others) - 10} more")
    for g, pids in foreign.items():
        print(f"PREFLIGHT: GPU {g} has foreign processes {pids}")
    if (others or foreign) and not args.force:
        sys.exit("preflight failed: the GPUs must be ours alone (--force to go ahead anyway)")
    blocks, extra = STUDIES[args.study](args)
    if args.max_shapes:
        keys = sorted(blocks)
        keys = keys[::max(1, len(keys) // args.max_shapes)][:args.max_shapes]
        blocks = {k: blocks[k] for k in keys}
    if args.max_points:
        blocks = {k: v[::max(1, len(v) // args.max_points)][:args.max_points] for k, v in blocks.items()}
    extra.update(unchecked=args.unchecked, max_shapes=args.max_shapes, max_points=args.max_points)
    blocks = {k: v for k, v in blocks.items() if v}
    slots, load = deal(blocks, len(gpus))
    n = sum(len(v) for v in blocks.values())
    if args.study in ("refit", "refit2", "worst10"):
        unchecked = [t for t in sweep.REFIT_TILES if not sweep.refit_checked(t)]
        pdp_unchecked = [f"{t} pdp{p}" for t, mx in sweep.PDP_MAX_PROGRAMS.items() if t in sweep.REFIT_TILES
                         for p in sweep.PDP_PROGRAMS if p <= mx and not sweep.refit_checked(t, p)]
        extra.update(unchecked_tiles=unchecked, unchecked_pdp=pdp_unchecked)
        if unchecked or pdp_unchecked:
            print(f"left out, not checked on this kernel: {', '.join(unchecked + pdp_unchecked)}")
        if extra.get("unchecked_raggedm"):
            print(f"left out of the memory-bound shapes, no ragged-M check: {', '.join(extra['unchecked_raggedm'])}")
    man = dict(study=args.study, prefix=prefix, created=time.strftime("%Y-%m-%d %H:%M:%S %z"),
               files=sweep.file_hashes(), triton=triton_info(), repo_head=git("rev-parse", "HEAD"),
               repo_dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
               v27_tracked=bool(git("ls-files", "v27_streamk_tile/matmul_kernel.py")), gpus=gpus,
               seed=SEEDS.get(args.study, sweep.REFIT_SEED), close_factor=CLOSE_FACTOR, iters="sweep.shape_iters",
               preflight=dict(others=others, foreign=foreign), points=n, shapes=len(blocks),
               estimate_gpu_hours_per_round=round(sum(load) / 3600, 2), slot_hours=[round(x / 3600, 2) for x in load],
               slots=slots, blocks=blocks, **extra)
    json.dump(man, open(path, "w"), indent=1)
    print(f"{path}: {len(blocks)} shapes, {n} points; estimate per round {sum(load) / 3600:.1f} GPU-hours, "
          f"{max(load) / 3600:.2f} h on {len(gpus)} GPUs (slots {', '.join(f'{x / 3600:.2f}' for x in load)} h)")
    print(f"frozen: {json.dumps(man['files'], indent=1)}")
    print(f"triton {man['triton']['version']} {man['triton']['commit']}{' (dirty)' if man['triton']['dirty'] else ''}"
          f"; repo {man['repo_head']}{' (dirty)' if man['repo_dirty'] else ''}"
          f"; v27 tracked by git: {man['v27_tracked']}")


# ---------------------------------------------------------------------------------------------------- shards


def round3_path(prefix):
    return os.path.join(RES, f"{prefix}_round3.json")


def shard_points(man, rnd, slot):
    """The points slot times in round rnd, block by block: shapes shuffled per (round, slot), each block shuffled per
    (round, shape); less the points whose warm compile failed."""
    G = len(man["gpus"])
    failed = warm_failures(man["prefix"])
    if rnd == 3:
        r3 = json.load(open(round3_path(man["prefix"])))
        keys = [k for k, s in r3["slots"].items() if s == slot]
        blocks = r3["points"]
    else:
        keys = [k for k, s in man["slots"].items() if (s + rnd - 1) % G == slot]
        blocks = man["blocks"]
    keys = sorted(keys)
    random.Random(f"{man['seed']}:{rnd}:{slot}").shuffle(keys)
    out = []
    for k in keys:
        pts = sorted(p for p in blocks[k] if p not in failed)
        random.Random(f"{man['seed']}:{rnd}:{k}").shuffle(pts)
        out += pts
    return out


def shard(args):
    man = load_manifest(args.prefix)
    sweep.check_frozen(manifest_path(args.prefix))
    gpu = man["gpus"][args.slot]
    assert os.environ.get("HIP_VISIBLE_DEVICES") == gpu, (
        f"slot {args.slot} is GPU {gpu}; run with HIP_VISIBLE_DEVICES={gpu}")
    pts = shard_points(man, args.round, args.slot)
    out = os.path.join(RES, f"{man['prefix']}_{args.round}_{gpu}.csv")
    est = sum(sweep.point_cost(p) for p in pts) / 3600
    print(f"round {args.round}, slot {args.slot} (GPU {gpu}): {len(pts)} points, estimate {est:.2f} h -> {out}",
          flush=True)
    sweep.run_points(pts, out, args.chunk, 0, args.point_timeout, extra={"round": args.round},
                     frozen=manifest_path(args.prefix), foreign_wait=args.foreign_wait)


# ---------------------------------------------------------------------------------------------------- analysis


def load_rounds(prefix):
    """({round: {point: row}} of the correct rows, [bad rows])."""
    out, bad = collections.defaultdict(dict), []
    for f in sorted(glob.glob(os.path.join(RES, f"{prefix}_[0-9]_*.csv"))):
        rnd = int(os.path.basename(f)[len(prefix) + 1:].split("_")[0])
        for r in csv.DictReader(open(f)):
            p = sweep.point_key(r)
            if r["correct"] == "True" and not r["error"] and r["us_median"]:
                r["us"] = float(r["us_median"])
                out[rnd][p] = r
            else:
                bad.append(r)
    return out, bad


def shape_of(p):
    return shape_key(sweep.parse_point(p)[2:])


def relative(timed, blocks, norm_sets):
    """{round: {point: rel}}: each round's time over the geometric mean, in that round, of the shape's normalising set
    (norm_sets[shape], else its whole block), restricted to the points timed in every round the shape was."""
    rel = collections.defaultdict(dict)
    for k, pts in blocks.items():
        rounds = [r for r in timed if any(p in timed[r] for p in pts)]
        base = [p for p in norm_sets.get(k, pts) if all(p in timed[r] for r in rounds)]
        if not base:
            continue
        for r in rounds:
            g = geomean([timed[r][p]["us"] for p in base])
            for p in pts:
                if p in timed[r]:
                    rel[r][p] = timed[r][p]["us"] / g
    return rel


def noise_of(rel, r1=1, r2=2):
    d = [abs(rel[r1][p] / rel[r2][p] - 1) for p in rel[r1] if p in rel[r2]]
    return d


def merged(rel, points):
    return {p: statistics.median([rel[r][p] for r in rel if p in rel[r]]) for p in points
            if any(p in rel[r] for r in rel)}


def by_tile(pts):
    out = collections.defaultdict(list)
    for p in pts:
        out[sweep.parse_point(p)[1]].append(p)
    return out


def explicit(p):
    return sweep.parse_point(p)[0] != "cap"


def decisions(blocks, val):
    """{shape: (best point per tile among explicit configurations, best tile)} from merged values."""
    out = {}
    for k, pts in blocks.items():
        tb = {}
        for t, ps in by_tile(pts).items():
            ps = [p for p in ps if explicit(p) and p in val]
            if ps:
                tb[t] = min(ps, key=val.get)
        if tb:
            out[k] = (tb, min(tb, key=lambda t: val[tb[t]]))
    return out


def close(args):
    man = load_manifest(args.prefix)
    blocks = man["blocks"]
    timed, _ = load_rounds(args.prefix)
    timed = {r: v for r, v in timed.items() if r in (1, 2)}
    assert set(timed) == {1, 2}, f"rounds 1 and 2 are needed, found {sorted(timed)}"
    rel = relative(timed, blocks, {})
    diffs = noise_of(rel)
    sig = pct(diffs, 0.9)
    val = merged(rel, [p for v in blocks.values() for p in v])
    lim = 1 + CLOSE_FACTOR * sig
    pick, why = {}, collections.Counter()
    for k, pts in blocks.items():
        sel = set()
        tb = {}
        for t, ps in by_tile(pts).items():
            ps = sorted((p for p in ps if explicit(p) and p in val), key=val.get)
            if ps:
                tb[t] = ps[0]
            if len(ps) > 1 and val[ps[1]] <= lim * val[ps[0]]:
                sel |= {ps[0], ps[1]}
                why["configuration"] += 1
        tiles = sorted(tb, key=lambda t: val[tb[t]])
        if len(tiles) > 1 and val[tb[tiles[1]]] <= lim * val[tb[tiles[0]]]:
            sel |= {tb[tiles[0]], tb[tiles[1]]}
            why["tile"] += 1
        dis = {p for p in pts if p in rel[1] and p in rel[2] and abs(rel[1][p] / rel[2][p] - 1) > sig}
        why["rounds disagree"] += len(dis)
        sel |= dis
        if sel:
            pick[k] = sorted(sel | {tb[t] for t in tiles[:3]})
    # A slot neither round used: (g + 2) % G or (g + 3) % G, whichever is less loaded.
    G = len(man["gpus"])
    load = [0.0] * G
    slots = {}
    cost = {k: sum(sweep.point_cost(p) for p in v) for k, v in pick.items()}
    for k in sorted(pick, key=lambda k: (-cost[k], k)):
        g = man["slots"][k]
        cands = [(g + 2) % G, (g + 3) % G] if G > 3 else [(g + 2) % G]
        s = min(cands, key=lambda i: (load[i], i))
        slots[k] = s
        load[s] += cost[k]
    n = sum(len(v) for v in pick.values())
    total = sum(len(v) for v in blocks.values())
    json.dump(dict(noise=sig, noise_median=statistics.median(diffs), noise_p99=pct(diffs, 0.99), noise_max=max(diffs),
                   pairs=len(diffs), why=why, share=n / total, points=pick, slots=slots,
                   slot_hours=[round(x / 3600, 2) for x in load]),
              open(round3_path(args.prefix), "w"), indent=1)
    print(f"noise (90th percentile of |rel_1 / rel_2 - 1| over {len(diffs)} points): {100 * sig:.2f}% "
          f"(median {100 * statistics.median(diffs):.2f}%, 99th {100 * pct(diffs, 0.99):.2f}%, "
          f"max {100 * max(diffs):.1f}%)")
    print(f"round 3: {n} points ({100 * n / total:.1f}%) on {len(pick)} shapes; close configurations "
          f"{why['configuration']} (shape, tile) pairs, close tiles {why['tile']} shapes, {why['rounds disagree']} "
          f"points whose rounds disagree; estimate {max(load) / 3600:.2f} h ({', '.join(f'{x / 3600:.2f}' for x in load)})")


def merge(args):
    man = load_manifest(args.prefix)
    blocks = man["blocks"]
    timed, bad = load_rounds(args.prefix)
    r3 = json.load(open(round3_path(args.prefix))) if os.path.exists(round3_path(args.prefix)) else None
    norm = r3["points"] if r3 and 3 in timed else {}
    rel = relative(timed, blocks, norm)
    allp = [p for v in blocks.values() for p in v]
    val = merged(rel, allp)
    dec = decisions(blocks, val)
    rows = []
    for k, pts in blocks.items():
        if k not in dec:
            continue
        tb, bt = dec[k]
        shape_best = val[tb[bt]]
        tile_best = {t: val[p] for t, p in tb.items()}
        for p in pts:
            if p not in val:
                continue
            rs = [r for r in sorted(timed) if p in timed[r]]
            r0 = dict(timed[rs[0]][p])
            us = statistics.median([timed[r][p]["us"] for r in rs])
            kind, tile, M, N, K = sweep.parse_point(p)
            relr = [rel[r][p] for r in rs if p in rel[r]]
            r0.update(us_median=round(us, 3), tflops=round(2 * M * N * K / (us * 1e-6) / 1e12, 1),
                      tbps=round(2 * (M * K + K * N + M * N) / (us * 1e-6) / 1e12, 3), rel=round(val[p], 5),
                      cfg_ratio=round(val[p] / tile_best[tile], 5) if tile in tile_best else "",
                      tile_ratio=round(tile_best[tile] / shape_best, 5) if tile in tile_best else "",
                      rounds=len(rs), gpus="/".join(timed[r][p]["gpu"] for r in rs),
                      spread=round(max(relr) / min(relr) - 1, 5),
                      **{f"us_r{r}": timed[r][p]["us"] if p in timed[r] else "" for r in (1, 2, 3)})
            rows.append(r0)
    fields = [f for f in sweep.FIELDS if f not in ("gpu", "round", "chunk", "pos", "ts")] + [
        "rel", "cfg_ratio", "tile_ratio", "rounds", "gpus", "spread", "us_r1", "us_r2", "us_r3"]
    out = os.path.join(RES, f"{args.prefix}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    report = os.path.join(RES, f"{args.prefix}_report.md")
    open(report, "w").write(make_report(man, timed, bad, rel, val, dec, r3, blocks))
    print(f"{len(rows)} points in {out}; report {report}")


# ---------------------------------------------------------------------------------------------------- report


def stats(xs, fmt="{:.2f}%", scale=100):
    if not xs:
        return "-"
    return " / ".join(fmt.format(scale * v) for v in (statistics.median(xs), pct(xs, 0.9), pct(xs, 0.99), max(xs)))


def coords(r):
    """A (shape, tile)'s coordinates: waves, log2(S / P) and k-steps (on the kernel's dimensions)."""
    P = sweep.v27_module().tile_geometry(r["tile"])["num_programs"]
    s = int(r["S"])
    return int(r["waves"]), (math.log2(s / P) if s else None), int(r["k_steps"])


def bin_label(x, edges, labels):
    for e, lab in zip(edges, labels):
        if x < e:
            return lab
    return labels[-1]


def make_report(man, timed, bad, rel, val, dec, r3, blocks):
    o = [f"# {man['study']} timing ({man['prefix']})\n"]
    o.append(f"Manifest `results/{man['prefix']}_manifest.json`, created {man['created']}: {man['shapes']} shapes, "
             f"{man['points']} points on GPUs {', '.join(man['gpus'])}; Triton {man['triton']['version']} "
             f"{man['triton']['commit']}; repository {man['repo_head']}. Frozen files (sha256, checked before every "
             "chunk):\n")
    for f, h in man["files"].items():
        o.append(f"- `{f}` {h}")
    rounds = sorted(timed)
    ts = [float(r["ts"]) for rr in timed.values() for r in rr.values() if r.get("ts")]
    o.append(f"\nRounds timed: {', '.join(f'{r} ({len(timed[r])} points)' for r in rounds)}"
             + (f", {time.strftime('%Y-%m-%d %H:%M', time.localtime(min(ts)))} to "
                f"{time.strftime('%H:%M', time.localtime(max(ts)))}" if ts else "") + ".")
    o.append(f"Rows not correct or with an error: {len(bad)}" + (":" if bad else "."))
    for r in bad[:20]:
        o.append(f"- round {r.get('round')} GPU {r.get('gpu')}: {sweep.point_key(r)} correct={r['correct']} "
                 f"{r['error']}")
    missing = [p for v in blocks.values() for p in v if p not in val]
    o.append(f"\nPoints with no correct timing: {len(missing)}.")

    o.append("\n## 1. Noise\n")
    d = noise_of(rel) if 1 in rel and 2 in rel else []
    sig = r3["noise"] if r3 else (pct(d, 0.9) if d else float("nan"))
    o.append(f"|rel_1 / rel_2 - 1| over {len(d)} points (median / 90th / 99th percentile / max): {stats(d)}. "
             f"The noise threshold sigma (90th percentile, from rounds 1 and 2) is **{100 * sig:.2f}%**.")
    o.append("FULL's repeats, for comparison: median 0.49%, 90th percentile 1.85%, max 10% (single timings on one GPU "
             "against another, GPU offsets included).\n")
    o.append("| group | points | median / 90th / 99th / max |")
    o.append("|---|---|---|")
    groups = collections.defaultdict(list)
    for p in rel.get(1, {}):
        if p in rel.get(2, {}):
            kind, tile, M, N, K = sweep.parse_point(p)
            x = abs(rel[1][p] / rel[2][p] - 1)
            groups[f"{regime(M)}, {tile_class(tile)}"].append(x)
            groups[f"iters {timed[1][p]['iters'] or '?'}"].append(x)
    for g in sorted(groups):
        o.append(f"| {g} | {len(groups[g])} | {stats(groups[g])} |")
    # Per-GPU offsets: a shape block's geometric-mean time on a GPU over its mean across that shape's rounds.
    off = collections.defaultdict(list)
    for k, pts in blocks.items():
        rs = [r for r in rounds if all(p in timed[r] for p in pts if p in val)]
        base = [p for p in pts if all(p in timed[r] for r in rs)]
        if len(rs) < 2 or not base:
            continue
        g = {r: geomean([timed[r][p]["us"] for p in base]) for r in rs}
        m = geomean(list(g.values()))
        for r in rs:
            off[timed[r][base[0]]["gpu"]].append(g[r] / m - 1)
    o.append("\nPer-GPU offset (a shape block's geometric-mean time on the GPU over its mean across the shape's rounds; "
             "within-block ratios cancel it):\n")
    o.append("| GPU | blocks | median | 10th | 90th |")
    o.append("|---|---|---|---|---|")
    for g in sorted(off):
        x = off[g]
        o.append(f"| {g} | {len(x)} | {100 * statistics.median(x):+.2f}% | {100 * pct(x, 0.1):+.2f}% | "
                 f"{100 * pct(x, 0.9):+.2f}% |")
    pos = collections.defaultdict(list)
    for r in rounds:
        for p, row in timed[r].items():
            if p in rel[r] and p in val and row.get("pos") not in (None, ""):
                b = bin_label(int(row["pos"]), (1, 5, 20), ("0", "1-4", "5-19", "20+"))
                pos[b].append(rel[r][p] / val[p] - 1)
    o.append("\nPosition in the chunk (each process's lead-in point is dropped): a timing over its point's merged "
             "value, median and 90th percentile of the absolute difference:\n")
    o.append("| position | timings | median | 90th percentile of abs |")
    o.append("|---|---|---|---|")
    for b in ("0", "1-4", "5-19", "20+"):
        if pos[b]:
            o.append(f"| {b} | {len(pos[b])} | {100 * statistics.median(pos[b]):+.2f}% | "
                     f"{100 * pct([abs(x) for x in pos[b]], 0.9):.2f}% |")

    o.append("\n## 2. Round 3\n")
    if r3:
        total = sum(len(v) for v in blocks.values())
        n3 = sum(len(v) for v in r3["points"].values())
        o.append(f"{n3} points ({100 * n3 / total:.1f}%) on {len(r3['points'])} shapes: (shape, tile) pairs whose best "
                 f"and runner-up configurations are within {CLOSE_FACTOR} sigma: {r3['why'].get('configuration', 0)}; "
                 f"shapes whose best and runner-up tiles are: {r3['why'].get('tile', 0)}; points whose rounds 1 and 2 "
                 f"differ by more than sigma: {r3['why'].get('rounds disagree', 0)}; plus each shape's best points at "
                 "its 3 best tiles as anchors.")
        if 3 in timed:
            rel12 = relative({r: timed[r] for r in (1, 2)}, blocks, r3["points"])
            v12 = merged(rel12, [p for v in blocks.values() for p in v])
            d12 = decisions(blocks, v12)
            ft = sum(1 for k in dec if k in d12 and d12[k][1] != dec[k][1])
            fc = sum(1 for k in dec if k in d12 for t in dec[k][0] if t in d12[k][0] and d12[k][0][t] != dec[k][0][t])
            agree = [abs(rel[3][p] / statistics.mean([rel[1][p], rel[2][p]]) - 1) for p in rel[3]
                     if p in rel[1] and p in rel[2]]
            o.append(f"\nRound 3 against the mean of rounds 1 and 2 (same normalisation): {stats(agree)} "
                     f"(median / 90th / 99th / max); within sigma for {100 * sum(x <= sig for x in agree) / max(1, len(agree)):.0f}%.")
            o.append(f"Decisions round 3 changed: the best tile of {ft} of {len(dec)} shapes, the best configuration "
                     f"of {fc} (shape, tile) pairs.")
        else:
            o.append("\nRound 3 has not been timed yet.")
    else:
        o.append("Not selected yet (`refit.py close`).")

    o.append("\n## 3. How close the winners are\n")
    o.append(f"After the merge: shapes whose runner-up tile, and (shape, tile) pairs whose runner-up configuration, is "
             f"within sigma ({100 * sig:.2f}%) and 2 sigma of the best.\n")
    o.append("| regime | shapes | runner-up tile within sigma | within 2 sigma | (shape, tile) pairs | runner-up "
             "configuration within sigma | within 2 sigma |")
    o.append("|---|---|---|---|---|---|---|")
    for g in ("compute-bound", "memory-bound"):
        ks = [k for k in dec if regime(parse_shape(k)[0]) == g]
        tr, cr = [], []
        for k in ks:
            tb, bt = dec[k]
            v = sorted(val[p] for p in tb.values())
            if len(v) > 1:
                tr.append(v[1] / v[0])
            for t, ps in by_tile(blocks[k]).items():
                c = sorted(val[p] for p in ps if explicit(p) and p in val)
                if len(c) > 1:
                    cr.append(c[1] / c[0])
        if ks:
            o.append(f"| {g} | {len(ks)} | {sum(x <= 1 + sig for x in tr)} | {sum(x <= 1 + 2 * sig for x in tr)} | "
                     f"{len(cr)} | {sum(x <= 1 + sig for x in cr)} | {sum(x <= 1 + 2 * sig for x in cr)} |")
    cap = [val[sweep.point_name("cap", "256x256x64", *parse_shape(k))] / val[dec[k][0][dec[k][1]]] for k in dec
           if sweep.point_name("cap", "256x256x64", *parse_shape(k)) in val]
    if cap:
        o.append(f"\nv26 as deployed (cap at 256x256x64) over each shape's best: median {statistics.median(cap):.3f}, "
                 f"90th percentile {pct(cap, 0.9):.3f}, worst {max(cap):.3f} ({len(cap)} shapes).")

    o.append("\n## 4. Coverage per tile class\n")
    o.append("(shape, tile) pairs per class in the tiles' own coordinates: waves of the tile's programs, leftover tiles "
             "S over programs P, and k-steps.\n")
    cov = collections.defaultdict(collections.Counter)
    for k, pts in blocks.items():
        M = parse_shape(k)[0]
        for t, ps in by_tile(pts).items():
            p = next((p for p in ps if any(p in timed[r] for r in rounds)), None)
            if p is None:
                continue
            r = next(timed[r][p] for r in rounds if p in timed[r])
            w, ls, ks = coords(r)
            c = f"{regime(M)}, {tile_class(t)}"
            cov[c]["pairs"] += 1
            cov[c]["w " + bin_label(w, (1, 4), ("0", "1-3", "4+"))] += 1
            cov[c]["S " + ("0" if ls is None else bin_label(ls, (-6, -4, -2), ("<1/64", "1/64-1/16", "1/16-1/4",
                                                                            "1/4-1")))] += 1
            cov[c]["k " + bin_label(ks, (9, 33, 129), ("<=8", "9-32", "33-128", ">128"))] += 1
    cols = ["pairs", "w 0", "w 1-3", "w 4+", "S 0", "S <1/64", "S 1/64-1/16", "S 1/16-1/4", "S 1/4-1", "k <=8",
            "k 9-32", "k 33-128", "k >128"]
    o.append("| class | " + " | ".join(cols) + " |")
    o.append("|---|" + "---|" * len(cols))
    for c in sorted(cov):
        o.append(f"| {c} | " + " | ".join(str(cov[c][x]) for x in cols) + " |")
    return "\n".join(o) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["init", "shard", "close", "merge"])
    ap.add_argument("--study", default="refit", choices=sorted(STUDIES))
    ap.add_argument("--prefix", help="results/<prefix>_*: default the study's name (init); shard, close, merge: refit")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--rule", help="heldout: a Python file with pick(M, N, K) -> (kind, tile)")
    ap.add_argument("--refit-prefix", default="refit", help="heldout: the merged REFIT results to take configurations from")
    ap.add_argument("--max-shapes", type=int, help="init: only this many shapes, evenly spread (a smoke test)")
    ap.add_argument("--max-points", type=int, help="init: at most this many points per shape (a smoke test)")
    ap.add_argument("--unchecked", action="store_true", help="init, refit: tiles without a check too (a smoke test)")
    ap.add_argument("--force", action="store_true", help="init: replace the manifest, ignore the preflight")
    ap.add_argument("--round", type=int, choices=[1, 2, 3])
    ap.add_argument("--slot", type=int)
    ap.add_argument("--chunk", type=int, default=60, help="points per rocprofv3 process (each costs 2 rocm-smi looks)")
    ap.add_argument("--point-timeout", type=int, default=600)
    ap.add_argument("--foreign-wait", type=int, default=1800)
    args = ap.parse_args()
    if args.mode == "init":
        init(args)
        return
    args.prefix = args.prefix or "refit"
    {"shard": shard, "close": close, "merge": merge}[args.mode](args)


if __name__ == "__main__":
    main()
