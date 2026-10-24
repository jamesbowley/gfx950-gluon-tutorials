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
"""v28's owner-ending variants against its base (v27's ending), timed in one block per shape on one GPU.

    source v28_streamk_owner/env.sh                     # HIP 1 = amd-smi GPU 0
    python v28_streamk_owner/owner_opt.py points        # the grid: shapes, points, estimate
    python v28_streamk_owner/owner_opt.py smoke --variant b
    python v28_streamk_owner/owner_opt.py warm
    python v28_streamk_owner/owner_opt.py run           # rounds 1-3, resumable (results/owner_opt/)
    python v28_streamk_owner/owner_opt.py report        # results/owner_opt/report.md

The variants are the archived experiment kernel's (STREAMK_OWNER_OPT; WORKLOG sections 1-9): run with
V28_MATMUL_MODULE=v28_streamk_owner.experiments.matmul_kernel_variants. v28 itself is their bs.

Shapes (SHAPES): per large tile, exactly waves * 256 + S tiles in the most nearly square M x N, for full waves 0
(S = 16, 64, 128, 192), 1 (S = 4, 32, 128, 224) and 4 (S = 8, 32, 128), at K = 2048 and 8192; plus the
1024x1024x7168 anchor at 128x128x128 and 128x128x64.

Points of a shape: ow2, ow3, ow4, owall, 2t and sp where sweep.valid takes them, and cap where the capped policy
runs the owner ending or spread, each in VARIANTS (2t, sp and an end-to-end cap without s and ps, which leave the
end-to-end build unchanged); cap otherwise once; rsall and rsall+bpw, the same code, as a null control.

Rounds 1 and 2 time every point. A block is a shape's points, shuffled, in one profiler process (with REF_POINT
before and after each chunk); every ratio is within a block. Round 3 re-times a variant and its base where the two
rounds' ratios differ by more than sigma, the 90th percentile of |rel_1 / rel_2 - 1| (rel: time over the block's
geometric mean in that round). The ratio of a variant is the median over rounds of its time over its base's.
"""

import argparse
import collections
import csv
import math
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRITON_CACHE = os.path.join(HERE, "tmp", "triton_owner_opt")
# Before anything imports Triton; the warm compile and the timing workers inherit it.
os.environ.setdefault("TRITON_CACHE_DIR", TRITON_CACHE)
sys.path.insert(0, HERE)
import sweep  # noqa: E402

OUT = os.path.join(HERE, "results", "owner_opt")
GPU_HIP = "1"
SEED = 2810
TILES = ("256x256x64", "256x128x64", "128x256x64", "128x128x64", "128x128x128")
GRID = {0: (16, 64, 128, 192), 1: (4, 32, 128, 224), 4: (8, 32, 128)}
KS = (2048, 8192)
ANCHORS = (("128x128x128", (1024, 1024, 7168)), ("128x128x64", (1024, 1024, 7168)))
OWNER_KINDS = ("ow2", "ow3", "ow4", "owall")
E2E_KINDS = ("2t", "sp")
VARIANTS = ("", "b", "bp", "bpw", "s", "ps")
E2E_VARIANTS = ("", "b", "bp", "bpw")
# The report's pairwise comparisons (later variant over earlier, "" = base), --steps.
STEPS = (("b", ""), ("bp", "b"), ("bpw", "bp"), ("s", ""), ("ps", "s"), ("s", "b"))
REF_POINT = "cap:256x256x64:4096x4096x4096"
REF_LIMIT = 0.05
CHUNK_POINTS = 48
FIELDS = sweep.FIELDS + ("rnd", "block", "is_ref")


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def shape_for(tile, T, K):
    """split_sweep.shape_for's 'square' geometry: (M, N, K) with exactly T tiles, the aspect closest to 1."""
    bm, bn = (int(x) for x in tile.split("x")[:2])
    opts = [(a * bm, T // a * bn) for a in range(1, T + 1) if T % a == 0]
    m, n = min(opts, key=lambda o: (abs(math.log(o[0] / o[1])), o[0]))
    return m, n, K


def shapes():
    """[(tile, waves, S, (M, N, K))]: the grid and the anchors."""
    out = []
    for tile in TILES:
        for w, ss in GRID.items():
            for s in ss:
                for k in KS:
                    out.append((tile, w, s, shape_for(tile, w * 256 + s, k)))
    for tile, shp in ANCHORS:
        sh = sweep.shape_info(tile, *shp)
        out.append((tile, sh["waves"], sh["S"], shp))
    return out


def cap_choice(tile, M, N, K):
    """(policy, fixup) the capped policy takes on the shape at the tile."""
    v28 = sweep.v27_module()
    geo = v28.tile_geometry(tile)
    bm, bn, bk = geo["block_m"], geo["block_n"], geo["block_k"]
    tiles = -(-M // bm) * -(-N // bn)
    k_pad = -(-K // (2 * bk)) * 2 * bk
    policy, fixup, _ = v28.capped_policy(tiles, k_pad // bk, geo["num_programs"])
    return policy, fixup


def block_points(tile, M, N, K):
    pts = []
    for kind in OWNER_KINDS:
        if sweep.valid(kind, tile, M, N, K):
            pts += [sweep.point_name(kind + (f"+{v}" if v else ""), tile, M, N, K) for v in VARIANTS]
    for kind in E2E_KINDS:
        if sweep.valid(kind, tile, M, N, K):
            pts += [sweep.point_name(kind + (f"+{v}" if v else ""), tile, M, N, K) for v in E2E_VARIANTS]
    policy, fixup = cap_choice(tile, M, N, K)
    cap_vars = (VARIANTS if policy == "one_tile" and fixup == "owner"
                else E2E_VARIANTS if policy in ("spread", "two_tile") else ("",))
    pts += [sweep.point_name("cap" + (f"+{v}" if v else ""), tile, M, N, K) for v in cap_vars]
    if sweep.valid("rsall", tile, M, N, K):
        pts += [sweep.point_name(k, tile, M, N, K) for k in ("rsall", "rsall+bpw")]
    return pts


def blocks():
    """{block id: [points]}, one block per shape."""
    return {f"{tile}:{M}x{N}x{K}": block_points(tile, M, N, K) for tile, w, s, (M, N, K) in shapes()}


def round_path(rnd):
    return os.path.join(OUT, f"r{rnd}.csv")


def read_rows(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def write_rows(out, rows):
    new = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def ref_values():
    return [float(r["us_median"]) for rnd in (1, 2, 3, 0) for r in read_rows(round_path(rnd))
            if r.get("is_ref") == "1" and r["us_median"]]


def time_blocks(bl, rnd, iters=0, foreign_wait=3600):
    """Time {block: [points]} not yet in the round's CSV (split_sweep.time_blocks): whole blocks per chunk, points
    shuffled in a block, blocks shuffled; REF_POINT before and after each chunk, which is timed again if the
    reference moves more than REF_LIMIT from the run's median or a foreign process is on the GPU after it."""
    out = round_path(rnd)
    done = {r["block"] for r in read_rows(out) if r.get("is_ref") == "0"}
    todo = [b for b in bl if b not in done and bl[b]]
    refs = ref_values()
    rng = random.Random(f"{SEED}|{rnd}")
    rng.shuffle(todo)
    total, chunk_id = len(todo), len({r["chunk"] for r in read_rows(out)})
    out_dir = os.path.join(sweep.TMP, f"sweep_owner_opt_r{rnd}")
    gpu = os.environ.get("HIP_VISIBLE_DEVICES")
    assert gpu == GPU_HIP, f"HIP_VISIBLE_DEVICES={gpu}: this study runs on HIP {GPU_HIP} (amd-smi GPU 0) only"
    while todo:
        chunk, n = [], 0
        while todo and (not chunk or n + len(bl[todo[0]]) <= CHUNK_POINTS):
            b = todo.pop(0)
            pts = list(bl[b])
            rng.shuffle(pts)
            chunk.append((b, pts))
            n += len(pts)
        flat = [REF_POINT, REF_POINT] + [p for _, pts in chunk for p in pts] + [REF_POINT]
        owner = [None, "REF", *[b for b, pts in chunk for _ in pts], "REF"]
        with sweep.gpu_lock(gpu):
            sweep.wait_for_gpu(gpu, foreign_wait, log)
            res, bad = sweep.timing(flat, out_dir, iters, 300)
            after = sweep.foreign_pids(gpu)
        if after:
            log(f"FOREIGN: pids {after} on the GPU after chunk {chunk_id}; timed again")
            todo = [b for b, _ in chunk] + todo
            continue
        if res is None:
            p = flat[bad]
            if p == REF_POINT:
                log(f"reference point failed in chunk {chunk_id}; timed again")
                todo = [b for b, _ in chunk] + todo
                continue
            b = owner[bad]
            log(f"{p} failed or hung; timed alone")
            r2, _ = sweep.timing([p, p], out_dir, iters, 300)
            row = r2[1] if r2 else sweep.row(p, None, error="failed or hung twice")
            row.update(rnd=rnd, block=b, is_ref=2 if not r2 else 0, gpu=gpu, chunk=f"{chunk_id}x", pos=0)
            write_rows(out, [row])
            bl[b] = [q for q in bl[b] if q != p]
            todo = [c for c, _ in chunk] + todo
            continue
        res = res[1:]
        now = [float(res[0]["us_median"]), float(res[-1]["us_median"])]
        med = statistics.median(refs) if len(refs) >= 4 else None
        if med and any(abs(x / med - 1) > REF_LIMIT for x in now):
            log(f"reference {now} us against the run's median {med:.1f} us in chunk {chunk_id}; timed again")
            todo = [b for b, _ in chunk] + todo
            continue
        refs += now
        rows = []
        for pos, (r, b) in enumerate(zip(res, owner[1:])):
            r.update(rnd=rnd, block=b if b != "REF" else "", is_ref=int(b == "REF"), gpu=gpu, chunk=chunk_id, pos=pos)
            rows.append(r)
        write_rows(out, rows)
        chunk_id += 1
        log(f"round {rnd}: {total - len(todo)} / {total} blocks")


def round_us(rnd):
    """{block: {point: us}} of a round's correct rows."""
    out = collections.defaultdict(dict)
    for r in read_rows(round_path(rnd)):
        if r.get("is_ref") != "0" or r["correct"] != "True" or r["error"] or not r["us_median"]:
            continue
        out[r["block"]][sweep.point_key(r)] = float(r["us_median"])
    return out


def base_of(p):
    """The base point of a variant point (the same point without +VARIANT), or None for a base point."""
    kind, tile, shp = p.split(":")[:3]
    k, var = sweep.owner_variant(kind)
    return f"{k}:{tile}:{shp}" if var else None


def rel(us):
    """{point: time over the block's geometric mean}."""
    g = math.exp(statistics.fmean(math.log(x) for x in us.values()))
    return {p: x / g for p, x in us.items()}


def sigma_of(r1, r2):
    d = []
    for b in r1:
        if b not in r2:
            continue
        common = set(r1[b]) & set(r2[b])
        if len(common) < 2:
            continue
        a = rel({p: r1[b][p] for p in common})
        c = rel({p: r2[b][p] for p in common})
        d += [abs(a[p] / c[p] - 1) for p in common]
    d.sort()
    return d[int(0.9 * (len(d) - 1))] if d else 0.02, d


def ratios(rounds):
    """{block: {variant point: [ratio over its base, per round that timed both]}}."""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for ru in rounds:
        for b, us in ru.items():
            for p, t in us.items():
                q = base_of(p)
                if q and q in us:
                    out[b][p].append(t / us[q])
    return out


def round3_blocks(bl, sigma):
    r1, r2 = round_us(1), round_us(2)
    out = {}
    for b, pts in bl.items():
        sel = set()
        for p in pts:
            q = base_of(p)
            if not q:
                continue
            if all(x in r1.get(b, {}) and x in r2.get(b, {}) for x in (p, q)):
                if abs((r1[b][p] / r1[b][q]) / (r2[b][p] / r2[b][q]) - 1) > sigma:
                    sel |= {p, q}
            else:
                sel |= {p, q}
        if sel:
            out[b] = sorted(sel)
    return out


def run(args):
    os.makedirs(OUT, exist_ok=True)
    bl = blocks()
    for rnd in (1, 2):
        log(f"round {rnd}: {sum(map(len, bl.values()))} points in {len(bl)} blocks")
        time_blocks({b: list(p) for b, p in bl.items()}, rnd, args.iters)
    sigma, _ = sigma_of(round_us(1), round_us(2))
    r3 = round3_blocks(bl, sigma)
    log(f"sigma {sigma:.4f}; round 3: {sum(map(len, r3.values()))} points in {len(r3)} blocks")
    time_blocks(r3, 3, args.iters)


def warm(args):
    pts = sorted({p for b in blocks().values() for p in b} | {REF_POINT})
    log(f"warm compile: {len(pts)} points")
    sweep.warm(pts, args.jobs, [GPU_HIP])


def smoke(args):
    """Base against one variant on the anchor (ow2, owall at 128x128x128) and a 256x256 owner shape, 3 rounds."""
    v = args.variant
    pts = []
    for kind, tile, shp in (("ow2", "128x128x128", "1024x1024x7168"), ("owall", "128x128x128", "1024x1024x7168"),
                            ("ow2", "128x128x64", "1024x1024x7168"), ("owall", "128x128x64", "1024x1024x7168"),
                            ("ow2", "256x256x64", "4096x4608x8192"), ("ow4", "256x256x64", "4096x4608x8192"),
                            ("ow2", "256x128x64", "3072x3072x8192"), ("sp", "128x128x128", "2048x1536x2048")):
        pts += [f"{kind}:{tile}:{shp}", f"{kind}+{v}:{tile}:{shp}"]
    out = os.path.join(OUT, f"smoke_{v}.csv")
    if os.path.exists(out):
        os.remove(out)
    os.makedirs(OUT, exist_ok=True)
    gpu = os.environ.get("HIP_VISIBLE_DEVICES")
    assert gpu == GPU_HIP, f"HIP_VISIBLE_DEVICES={gpu}: this study runs on HIP {GPU_HIP} only"
    res = collections.defaultdict(list)
    for rnd in range(3):
        rng = random.Random(f"smoke|{v}|{rnd}")
        order = list(pts)
        rng.shuffle(order)
        with sweep.gpu_lock(gpu):
            rows, bad = sweep.timing(order[:1] + order, os.path.join(sweep.TMP, "sweep_owner_smoke"), 0, 300)
        assert rows, f"smoke timing failed at {order[max(bad - 1, 0)]}"
        for r in rows[1:]:
            assert r["correct"], f"wrong result: {sweep.point_key(r)}"
            res[sweep.point_key(r)].append(float(r["us_median"]))
    print(f"| point | base us | +{v} us | ratio (median of 3) |")
    print("|---|---|---|---|")
    for p in pts[::2]:
        q = p.replace(":", f"+{v}:", 1)
        rs = sorted(a / b for a, b in zip(res[q], res[p]))
        print(f"| {p} | {statistics.median(res[p]):.2f} | {statistics.median(res[q]):.2f} | {rs[1]:.3f} |")


def points(args):
    bl = blocks()
    total = sum(map(len, bl.values()))
    est = sum(sweep.point_cost(p) for b in bl.values() for p in b) if hasattr(sweep, "point_cost") else 0
    for tile, w, s, shp in shapes():
        b = f"{tile}:{'x'.join(map(str, shp))}"
        print(f"{tile} w{w} S={s} {b.split(':')[1]}: {len(bl[b])} points, cap {cap_choice(tile, *shp)}")
    print(f"{len(bl)} blocks, {total} points per round; estimated cost {est / 3600:.2f} h per round (sweep.point_cost)")


# ------------------------------------------------------------------------------------------------ report


def gmean(xs):
    return math.exp(statistics.fmean(math.log(x) for x in xs)) if xs else float("nan")


def report(args):
    bl = blocks()
    rounds = [round_us(r) for r in (1, 2, 3)]
    sigma, diffs = sigma_of(rounds[0], rounds[1])
    rat = ratios(rounds)
    info = {f"{tile}:{M}x{N}x{K}": (tile, w, s, (M, N, K)) for tile, w, s, (M, N, K) in shapes()}
    rows = []
    for b, ps in rat.items():
        tile, w, s, shp = info[b]
        for p, rs in ps.items():
            kind, var = sweep.owner_variant(p.split(":")[0])
            rows.append(dict(block=b, tile=tile, w=w, S=s, K=shp[2], kind=kind, var=var, ratio=statistics.median(rs),
                             n=len(rs)))
    n_r = {r: sum(len(v) for v in rounds[r - 1].values()) for r in (1, 2, 3)}
    refs = ref_values()
    L = [f"# v28 owner-ending variants against v27's ending (base), {time.strftime('%Y-%m-%d')}", ""]
    L.append(f"HIP {GPU_HIP} (amd-smi GPU 0, PCI 0000:05:00.0). {len(bl)} blocks (shapes); timings per round: "
             f"{n_r[1]}, {n_r[2]}, {n_r[3]}. Reference ({REF_POINT}): {min(refs):.1f}-{max(refs):.1f} us. "
             f"Noise sigma (90th percentile of |rel_1 / rel_2 - 1|): {100 * sigma:.2f}% "
             f"(median {100 * statistics.median(diffs):.2f}%). A ratio is a variant's time over its base's in the "
             "same block, the median over rounds; below 1 is faster.")
    L.append("")
    null = [r["ratio"] for r in rows if r["kind"] == "rsall"]
    if null:
        L.append(f"**Null control** (rsall+bpw over rsall, the same code): {len(null)} shapes, geomean "
                 f"{gmean(null):.4f}, range {min(null):.3f}-{max(null):.3f}, outside +-sigma: "
                 f"{sum(abs(x - 1) > sigma for x in null)}.")
        L.append("")

    def family(kind):
        return "one-tile owner" if kind in OWNER_KINDS else "2t / sp" if kind in E2E_KINDS else kind

    def summary(sel, title):
        L.append(f"## {title}")
        L.append("")
        L.append("Geomean ratio [best, worst] (points faster / slower than base by more than sigma).")
        L.append("")
        hdr = "| tile | waves | family | points | " + " | ".join(v for v in VARIANTS[1:]) + " |"
        L.append(hdr)
        L.append("|" + "---|" * (hdr.count("|") - 1))
        keys = sorted({(r["tile"], r["w"], family(r["kind"])) for r in sel},
                      key=lambda k: (TILES.index(k[0]), k[1], k[2]))
        for t, w, f in keys:
            cells, npts = [], 0
            for v in VARIANTS[1:]:
                xs = [r["ratio"] for r in sel if (r["tile"], r["w"], family(r["kind"])) == (t, w, f) and r["var"] == v]
                npts = max(npts, len(xs))
                if not xs:
                    cells.append("")
                    continue
                cells.append(f"{gmean(xs):.3f} [{min(xs):.3f}, {max(xs):.3f}] "
                             f"({sum(x < 1 - sigma for x in xs)}/{sum(x > 1 + sigma for x in xs)})")
            L.append(f"| {t} | {w} | {f} | {npts} | " + " | ".join(cells) + " |")
        L.append("")

    owner_rows = [r for r in rows if r["kind"] != "rsall"]
    summary(owner_rows, "By tile, wave class and family")
    L.append("## By tile and variant, every owner point")
    L.append("")
    hdr = "| tile | " + " | ".join(VARIANTS[1:]) + " |"
    L.append(hdr)
    L.append("|" + "---|" * (hdr.count("|") - 1))
    verdict = {}
    for t in TILES:
        cells = []
        for v in VARIANTS[1:]:
            xs = [r["ratio"] for r in owner_rows if r["tile"] == t and r["var"] == v]
            cells.append(f"{gmean(xs):.3f} [{min(xs):.3f}, {max(xs):.3f}] ({sum(x < 1 - sigma for x in xs)}/"
                         f"{sum(x > 1 + sigma for x in xs)})" if xs else "")
            verdict[(t, v)] = xs
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## By owner kind (one-tile n) and variant, 128x128 tiles and 256x256x64")
    L.append("")
    hdr = "| tile | kind | points | " + " | ".join(VARIANTS[1:]) + " |"
    L.append(hdr)
    L.append("|" + "---|" * (hdr.count("|") - 1))
    for t in TILES:
        for k in OWNER_KINDS + E2E_KINDS + ("cap",):
            cells, npts = [], 0
            for v in VARIANTS[1:]:
                xs = [r["ratio"] for r in owner_rows if r["tile"] == t and r["kind"] == k and r["var"] == v]
                npts = max(npts, len(xs))
                cells.append(f"{gmean(xs):.3f} [{min(xs):.3f}, {max(xs):.3f}]" if xs else "")
            if npts:
                L.append(f"| {t} | {k} | {npts} | " + " | ".join(cells) + " |")
    L.append("")
    # The anchor: absolute TFLOPS per round's median time.
    L.append("## Anchor 1024x1024x7168 (TFLOPS, median over rounds)")
    L.append("")
    L.append("hipBLASLt 1.2.2 at the same split (2026-10-19, cold harness, physical GPU 1): 546.0 on 4 programs "
             "per tile, 407.0 on 2. These are rocprof back-to-back timings, so the comparison is approximate.")
    L.append("")
    hdr = "| tile | kind | " + " | ".join(v or "base" for v in VARIANTS) + " |"
    L.append(hdr)
    L.append("|" + "---|" * (hdr.count("|") - 1))
    for tile, (M, N, K) in ANCHORS:
        b = f"{tile}:{M}x{N}x{K}"
        for k in OWNER_KINDS + ("cap", "rsall"):
            cells = []
            for v in VARIANTS:
                p = sweep.point_name(k + (f"+{v}" if v else ""), tile, M, N, K)
                ts = [ru[b][p] for ru in rounds if p in ru.get(b, {})]
                cells.append(f"{2 * M * N * K / (statistics.median(ts) * 1e-6) / 1e12:.1f}" if ts else "")
            if any(cells):
                L.append(f"| {tile} | {k} | " + " | ".join(cells) + " |")
    L.append("")
    # Decision rule.
    L.append("## Verdict per variant")
    L.append("")
    L.append("Accept if: the owner points of the 128x128 tiles improve (geomean below 1); no 256x256x64 point is worse "
             f"than base by more than 2 sigma ({1 + 2 * sigma:.4f}); 256x256x64's geomean is no more than 0.5% above "
             "base. The plan wrote the last as 'within 0.5%'; it is a regression guard, so a faster 256x256x64 passes.")
    L.append("")
    L.append("| variant | 128x128 geomean | 256x256x64 geomean | 256x256x64 worst | 256x128 / 128x256 geomean | "
             "accept |")
    L.append("|---|---|---|---|---|---|")
    for v in VARIANTS[1:]:
        small = verdict.get(("128x128x128", v), []) + verdict.get(("128x128x64", v), [])
        big = verdict.get(("256x256x64", v), [])
        rect = verdict.get(("256x128x64", v), []) + verdict.get(("128x256x64", v), [])
        ok = (small and gmean(small) < 1 and (not big or (max(big) <= 1 + 2 * sigma and gmean(big) <= 1.005)))
        fmt = lambda xs, f=gmean: f"{f(xs):.3f}" if xs else "not timed"  # noqa: E731
        L.append(f"| {v} | {fmt(small)} | {fmt(big)} | {fmt(big, max)} | {fmt(rect)} | {'yes' if ok else 'no'} |")
    L.append("")
    # Each change on top of the previous one in the chain: within-block, per-round ratios of the two variants.
    L.append("## Each change on top of the previous one")
    L.append("")
    L.append("Ratio of the later variant over the earlier, in the same blocks (median over rounds), geomean [best, "
             "worst] (faster / slower by more than sigma), over the points that have both.")
    L.append("")
    steps = STEPS
    hdr = "| tile | family | " + " | ".join(f"{a or 'base'} / {b or 'base'}" for a, b in steps) + " |"
    L.append(hdr)
    L.append("|" + "---|" * (hdr.count("|") - 1))
    pair = collections.defaultdict(lambda: collections.defaultdict(list))
    for ru in rounds:
        for b, us in ru.items():
            for p, t in us.items():
                kind, var = sweep.owner_variant(p.split(":")[0])
                if kind == "rsall":
                    continue
                rest = p.split(":", 1)[1]
                for a, c in steps:
                    if var == a:
                        q = f"{kind}{'+' + c if c else ''}:{rest}"
                        if q in us:
                            pair[(info[b][0], family(kind), a, c)][(b, kind)].append(t / us[q])
    for t in TILES:
        for f in ("one-tile owner", "2t / sp", "cap"):
            cells = []
            for a, c in steps:
                xs = [statistics.median(v) for v in pair[(t, f, a, c)].values()]
                cells.append(f"{gmean(xs):.3f} [{min(xs):.3f}, {max(xs):.3f}] ({sum(x < 1 - sigma for x in xs)}/"
                             f"{sum(x > 1 + sigma for x in xs)})" if xs else "")
            L.append(f"| {t} | {f} | " + " | ".join(cells) + " |")
    L.append("")
    # By programs per tile: where the poll (bp / b) and the compile-time peer loop (s / base, s / b) pay.
    L.append("## By programs per tile")
    L.append("")
    L.append("One-tile owner points by n (owall: n = min(256 // S, pairs)), 2t and sp; geomean of the within-block "
             "ratio (median over rounds). Tile groups: 128x128 (BK 64 and 128), rect (256x128, 128x256), 256x256.")
    L.append("")
    cols = [f"{a or 'base'} / {c or 'base'}" for a, c in STEPS]
    L.append("| tiles | kind (n) | points | " + " | ".join(cols) + " |")
    L.append("|" + "---|" * (3 + len(cols)))
    grp = collections.defaultdict(lambda: collections.defaultdict(list))
    for b, (tile, w, s, (M, N, K)) in info.items():
        g = "128x128" if tile.startswith("128x128") else "256x256" if tile.startswith("256x256") else "rect"
        for kind in OWNER_KINDS + E2E_KINDS:
            if not sweep.valid(kind, tile, M, N, K):
                continue
            if kind == "owall":
                sh = sweep.shape_info(tile, M, N, K)
                label, order = f"owall (n = {min(256 // sh['S'], sh['pairs'])})", min(256 // sh["S"], sh["pairs"]) + .5
            else:
                label, order = kind, {"ow2": 2, "ow3": 3, "ow4": 4, "sp": 1000, "2t": 1001}[kind]
            rest = f"{tile}:{M}x{N}x{K}"
            for col, (a, c) in zip(cols, ((f"+{a}" if a else "", f"+{c}" if c else "") for a, c in STEPS)):
                xs = [ru[b][f"{kind}{a}:{rest}"] / ru[b][f"{kind}{c}:{rest}"] for ru in rounds
                      if f"{kind}{a}:{rest}" in ru.get(b, {}) and f"{kind}{c}:{rest}" in ru.get(b, {})]
                if xs:
                    grp[(g, order, label)][col].append(statistics.median(xs))
    for key in sorted(grp, key=lambda k: (("128x128", "rect", "256x256").index(k[0]), k[1])):
        d = grp[key]
        cells = [f"{gmean(d[c]):.3f}" if d[c] else "" for c in cols]
        L.append(f"| {key[0]} | {key[2]} | {max(len(d[c]) for c in cols)} | " + " | ".join(cells) + " |")
    L.append("")
    # Detail.
    L.append("## Every point")
    L.append("")
    hdr = "| tile | waves | S | K | kind | " + " | ".join(VARIANTS[1:]) + " |"
    L.append(hdr)
    L.append("|" + "---|" * (hdr.count("|") - 1))
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["tile"], r["w"], r["S"], r["K"], r["kind"])][r["var"]] = r["ratio"]
    for key in sorted(by, key=lambda k: (TILES.index(k[0]), k[1], k[2], k[3], k[4])):
        L.append("| " + " | ".join(map(str, key)) + " | "
                 + " | ".join(f"{by[key][v]:.3f}" if v in by[key] else "" for v in VARIANTS[1:]) + " |")
    out = os.path.join(OUT, "report.md")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:60]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["points", "smoke", "warm", "run", "report"])
    ap.add_argument("--variant", default="b")
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--iters", type=int, default=0, help="launches per point; 0: sweep.shape_iters")
    ap.add_argument("--tiles", help="comma-separated tiles (default: TILES)")
    ap.add_argument("--variants", help="comma-separated STREAMK_OWNER_OPT variants after base (default: VARIANTS)")
    ap.add_argument("--steps", help="comma-separated later/earlier variant pairs for the report, e.g. b/ba,s/b")
    ap.add_argument("--out", help="results directory (default: results/owner_opt)")
    args = ap.parse_args()
    global TILES, VARIANTS, E2E_VARIANTS, STEPS, OUT
    if args.tiles:
        TILES = tuple(args.tiles.split(","))
    if args.variants:
        VARIANTS = ("",) + tuple(args.variants.split(","))
        # The compile-time peer loop (s) leaves the end-to-end splits' build unchanged.
        E2E_VARIANTS = tuple(v for v in VARIANTS if "s" not in v)
    if args.steps:
        STEPS = tuple(tuple(x.split("/")) for x in args.steps.split(","))
    if args.out:
        OUT = os.path.join(HERE, "results", args.out)
    if any(VARIANTS) and os.environ.get("V28_MATMUL_MODULE") != "v28_streamk_owner.experiments.matmul_kernel_variants":
        sys.exit("owner_opt.py times STREAMK_OWNER_OPT variants, which only the archived experiment kernel has: set "
                 "V28_MATMUL_MODULE=v28_streamk_owner.experiments.matmul_kernel_variants")
    {"points": points, "smoke": smoke, "warm": warm, "run": run, "report": report}[args.mode](args)


if __name__ == "__main__":
    main()
