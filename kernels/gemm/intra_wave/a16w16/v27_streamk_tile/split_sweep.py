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
"""Per-tile split tables on one GPU: the best stream-K method of each tile as a function of its own leftover tiles
S, k-steps and full waves (the two-stage plan's first stage).

    source v27_streamk_tile/env.sh                         # HIP 2 = rocm-smi 2 (physical GPU 2)
    python v27_streamk_tile/split_sweep.py plan            # cells, points, estimate; nothing launched
    python v27_streamk_tile/split_sweep.py run             # every stage, resumable (results/split/)
    python v27_streamk_tile/split_sweep.py report          # tables.json, tables.md from what is timed

A cell is (tile, wave class, S, k-steps): one real shape with exactly waves * P + S tiles at the tile and K = k-steps
* BK, never a cut-down last wave. Its block (every method at that shape) is timed back to back, in a shuffled order,
in one profiler process; every comparison is within a block, so nothing is compared across runs, days or GPUs.

Stages, each a racing sequence (round 1: every point; round 2: the points within leader * (1 + Z sigma + DELTA),
sigma the noise model's 90th percentile for the point's tile class and duration; round 3: the two leaders if still
within sigma):
- A: the coarse grid (S_GRID, K_TILES, WAVE_CLASSES), the methods of methods();
- B1, B2: boundary refinement: a cell between two neighbours (in S or k) whose picks cost each other more than sigma;
- N: the explicit n between the coarse neighbours of a cell's best one-tile n, timed with its leader and runner-up.
Every chunk starts and ends with REF_POINT (after a dropped lead-in), and is timed again if the reference moves more
than REF_LIMIT from the run's median, or a foreign process is on the GPU after it.
"""

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import refit  # noqa: E402
import sweep  # noqa: E402

OUT = os.environ.get("SPLIT_OUT", os.path.join(HERE, "results", "split"))
TRITON_CACHE = os.path.join(HERE, "tmp", "triton_split")
SEED = 2710
GPU_HIP, GPU_SMI = "2", "2"

QUAD_TILES = ("256x256x64", "256x128x64", "128x256x64", "128x128x64", "128x128x128", "64x128x128", "64x64x128")
SMALL_TILES = ("1x16x256", "16x16x256", "16x16x128", "16x32x128", "32x32x128", "32x128x128")
# A wave class takes the first wave count, in this order, whose tiles factor into a shape within the bounds.
WAVE_CLASSES = {0: (0,), 1: (1, 2, 3), 4: (4, 5, 6, 7)}
K_TILES = (4, 8, 16, 32, 64, 128, 256, 512)
K_RANGE = {"quadrant": (512, 32768), "small": (1024, 32768)}
DIM_MAX = 131072
SMALL_M_MAX = 448
COARSE_N = (2, 4, 8, 16, 32)
REFINE_N = (3, 5, 6, 7, 10, 12, 24)
OW_N = (2, 3, 4)
SP_MIN_SHARE = 0.25
Z, DELTA = 2.0, 0.01
SIGMA_DEFAULT = 0.0189
REF_POINT = "cap:256x256x64:4096x4096x4096"
REF_LIMIT = 0.05
CHUNK_POINTS = 48
FROZEN = sweep.FROZEN_FILES + ("v27_streamk_tile/split_sweep.py",)
SIMPLE = ("dp", "cap", "pdp", "pdp512", "rs2", "rs3", "rs4", "rs5", "rs6", "rs7", "rs8", "rs10", "rs12", "rs16",
          "rs24", "rs32", "rsall", "ow2", "ow3", "ow4", "owall", "2t", "sp")


def s_grid(P):
    return sorted({2 ** i for i in range(int(math.log2(P // 2)) + 1)} | {P * 5 // 8, P * 3 // 4, P * 7 // 8, P - 1})


def tclass(tile):
    return "small" if tile in SMALL_TILES or sweep.v27_module().tile_geometry(tile)["small"] else "quadrant"


def geom(tile):
    g = sweep.v27_module().tile_geometry(tile)
    return g["block_m"], g["block_n"], g["block_k"], g["num_programs"], g["swap"], g["small"]


def divisors(n):
    small = [d for d in range(1, int(math.isqrt(n)) + 1) if n % d == 0]
    return sorted(set(small + [n // d for d in small]))


def shape_for(tile, T, K, which):
    """(M, N, K) with exactly T tiles at the tile, or None. Quadrant: 'square' (aspect closest to 1) or 'skinny'
    (the most elongated within DIM_MAX, M small if which == 'skinny-m'); small tiles: rows of tiles a with a * BM <=
    SMALL_M_MAX, 'row' (a = 1) or 'square' (the largest such a)."""
    bm, bn, bk, P, swap, small = geom(tile)
    opts = []
    for a in divisors(T):
        m, n = a * bm, (T // a) * bn
        if m > DIM_MAX or n > DIM_MAX or (small and m > SMALL_M_MAX):
            continue
        opts.append((a, m, n))
    if not opts:
        return None
    if small:
        a, m, n = opts[0] if which == "row" else opts[-1]
    elif which == "square":
        a, m, n = min(opts, key=lambda o: (abs(math.log(o[1] / o[2])), o[1]))
    else:
        a, m, n = min(opts, key=lambda o: o[1]) if which == "skinny-m" else max(opts, key=lambda o: o[1])
    M, N = (n, m) if swap else (m, n)
    return M, N, K


def cell_shape(tile, wc, S, k, alt):
    """The cell's shape: the first wave count of the class that fits; the geometry alternates over the grid (alt)."""
    bm, bn, bk, P, swap, small = geom(tile)
    K = k * bk
    lo, hi = K_RANGE["small" if small else "quadrant"]
    if not lo <= K <= hi:
        return None
    which = (("row", "square") if small else ("square", "skinny-m", "square", "skinny-n"))[alt % (2 if small else 4)]
    for w in WAVE_CLASSES[wc]:
        sh = shape_for(tile, w * P + S, K, which)
        if sh:
            return sh
    return None


def methods(tile, M, N, K, wc, n_set=COARSE_N):
    bm, bn, bk, P, swap, small = geom(tile)
    sh = sweep.shape_info(tile, M, N, K)
    kinds = ["dp", "pdp", "cap"]
    if sweep.PDP_MAX_PROGRAMS.get(tile, 256) >= 512:
        kinds.append("pdp512")
    if wc > 0:
        kinds.append("2t")
    if sh["S"] >= SP_MIN_SHARE * P:
        kinds.append("sp")
    kinds += [f"rs{n}" for n in n_set] + ["rsall"] + [f"ow{n}" for n in OW_N] + ["owall"]
    return [k for k in kinds if sweep.valid(k, tile, M, N, K)]


# ------------------------------------------------------------------------------------------------ cells


def coarse_cells():
    """[cell]: cell = dict(id, tile, wc, S, k, M, N, K)."""
    cells = []
    for tile in QUAD_TILES + SMALL_TILES:
        P = geom(tile)[3]
        for wi, wc in enumerate(WAVE_CLASSES):
            for si, S in enumerate(s_grid(P)):
                for ki, k in enumerate(K_TILES):
                    sh = cell_shape(tile, wc, S, k, si + ki + wi)
                    if sh:
                        cells.append(make_cell(tile, wc, S, k, sh, "A"))
    return cells


def make_cell(tile, wc, S, k, sh, stage):
    M, N, K = sh
    return dict(id=f"{tile}|w{wc}|S{S}|k{k}|{M}x{N}x{K}", tile=tile, wc=wc, S=S, k=k, M=M, N=N, K=K, stage=stage)


def block_points(cell, kinds):
    return [sweep.point_name(kd, cell["tile"], cell["M"], cell["N"], cell["K"]) for kd in kinds]


# ------------------------------------------------------------------------------------------------ noise model


def load_sigma():
    """{(class, duration bin): 90th percentile}: this run's round 1 / round 2 differences where there are enough,
    else Phase 0's (REFIT's), else SIGMA_DEFAULT."""
    sig = {}
    p0 = os.path.join(HERE, "results", "phase0.json")
    if os.path.exists(p0):
        for k, d in json.load(open(p0)).get("noise", {}).items():
            cls, b = k.split(" ", 1)
            sig[(cls, b)] = d["p90"]
    own = os.path.join(OUT, "noise.json")
    if os.path.exists(own):
        for k, d in json.load(open(own)).items():
            cls, b = k.split(" ", 1)
            if d["n"] >= 200:
                sig[(cls, b)] = d["p90"]
    return sig


DUR_EDGES = (15, 30, 60, 150, 500)


def dur_bin(us):
    for e in DUR_EDGES:
        if us < e:
            return f"<{e}"
    return f">={DUR_EDGES[-1]}"


# ------------------------------------------------------------------------------------------------ timing


def file_hashes():
    return {f: hashlib.sha256(open(os.path.join(sweep.A16, f), "rb").read()).hexdigest() for f in FROZEN}


def manifest():
    path = os.path.join(OUT, "manifest.json")
    if os.path.exists(path):
        return json.load(open(path))
    os.makedirs(OUT, exist_ok=True)
    m = dict(created=time.strftime("%Y-%m-%d %H:%M:%S %z"), files=file_hashes(), triton=refit.triton_info(),
             gpu_hip=GPU_HIP, gpu_smi=GPU_SMI, seed=SEED, z=Z, delta=DELTA, ref=REF_POINT, ref_limit=REF_LIMIT,
             quad_tiles=QUAD_TILES, small_tiles=SMALL_TILES, k_tiles=K_TILES, wave_classes=WAVE_CLASSES)
    json.dump(m, open(path, "w"), indent=1)
    return m


def check_frozen(m):
    now = file_hashes()
    changed = [f for f, h in m["files"].items() if now.get(f) != h]
    if changed:
        log(f"FROZEN: {', '.join(changed)} changed since the manifest; stopping")
        sys.exit(3)


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def round_path(stage, rnd):
    return os.path.join(OUT, f"{stage}_r{rnd}.csv")


FIELDS = sweep.FIELDS + ("stage", "rnd", "block", "is_ref")


def read_rows(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def ref_values():
    """Every reference timing of the run so far."""
    return [float(r["us_median"]) for f in sorted(os.listdir(OUT)) if f.endswith(".csv")
            for r in read_rows(os.path.join(OUT, f)) if r.get("is_ref") == "1" and r["us_median"]]


def time_blocks(blocks, stage, rnd, m, iters=0, foreign_wait=3600):
    """Time {block id: [points]} not yet in the round's CSV: whole blocks per chunk, points shuffled in a block,
    blocks shuffled; REF_POINT before and after each chunk's blocks."""
    out = round_path(stage, rnd)
    prev = read_rows(out)
    done = {r["block"] for r in prev if r.get("is_ref") == "0"}
    for r in prev:
        if r.get("is_ref") == "2":
            blocks[r["block"]] = [q for q in blocks.get(r["block"], []) if q != sweep.point_key(r)]
    todo = [b for b in blocks if b not in done and blocks[b]]
    refs = ref_values()
    rng = random.Random(f"{SEED}|{stage}|{rnd}")
    rng.shuffle(todo)
    total, chunk_id = len(todo), len({r["chunk"] for r in read_rows(out)})
    tag = f"split_{stage}_r{rnd}"
    out_dir = os.path.join(sweep.TMP, f"sweep_{tag}")
    gpu = os.environ.get("HIP_VISIBLE_DEVICES", GPU_HIP)
    assert gpu == GPU_HIP, f"HIP_VISIBLE_DEVICES={gpu}, the manifest's GPU is HIP {GPU_HIP}"
    while todo:
        check_frozen(m)
        chunk, n = [], 0
        while todo and (not chunk or n + len(blocks[todo[0]]) <= CHUNK_POINTS):
            b = todo.pop(0)
            pts = list(blocks[b])
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
            row.update(stage=stage, rnd=rnd, block=b, is_ref=2 if not r2 else 0, gpu=gpu, chunk=f"{chunk_id}x", pos=0)
            write_rows(out, [row])
            blocks[b] = [q for q in blocks[b] if q != p]
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
            r.update(stage=stage, rnd=rnd, block=b if b != "REF" else "", is_ref=int(b == "REF"), gpu=gpu,
                     chunk=chunk_id, pos=pos)
            rows.append(r)
        write_rows(out, rows)
        chunk_id += 1
        log(f"{stage} round {rnd}: {total - len(todo)} / {total} blocks")
    import shutil

    shutil.rmtree(out_dir, ignore_errors=True)


def write_rows(out, rows):
    new = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def warm(points, jobs):
    todo = sorted({p for p in points if sweep.parse_point(p)[0] != "dp"})
    if todo:
        log(f"warm compile: {len(todo)} points")
        sweep.warm(todo, jobs, [GPU_HIP])


# ------------------------------------------------------------------------------------------------ racing


def round_us(stage, rnd):
    """{point: us} of a round's correct rows."""
    out = {}
    for r in read_rows(round_path(stage, rnd)):
        if r.get("is_ref") == "1" or r["correct"] != "True" or r["error"] or not r["us_median"]:
            continue
        out[sweep.point_key(r)] = float(r["us_median"])
    return out


def relative(blocks, rounds):
    """{point: merged rel}: per block, each round's time over the geometric mean, in that round, of the block's points
    timed in every round the block was timed (refit.relative), then the median over rounds."""
    timed = {r: u for r, u in rounds.items() if u}
    rel = refit.relative({r: {p: {"us": v} for p, v in u.items()} for r, u in timed.items()}, blocks, {})
    out = {}
    for b, pts in blocks.items():
        for p in pts:
            vs = [rel[r][p] for r in rel if p in rel[r]]
            if vs:
                out[p] = statistics.median(vs)
    return out, rel


def simplicity(p):
    k = sweep.parse_point(p)[0]
    return SIMPLE.index(k) if k in SIMPLE else len(SIMPLE)


def pick(pts, val):
    """The lowest merged value; values within DELTA of it are a tie, broken by SIMPLE's order."""
    pts = [p for p in pts if p in val]
    if not pts:
        return None
    best = min(val[p] for p in pts)
    return min((p for p in pts if val[p] <= best * (1 + DELTA)), key=lambda p: (simplicity(p), val[p]))


def race(stage, blocks, m, jobs):
    """Rounds 1-3 of a stage over {block: [points]}; returns ({point: merged rel}, {block: survivors})."""
    warm([p for v in blocks.values() for p in v], jobs)
    time_blocks(dict(blocks), stage, 1, m)
    us1 = round_us(stage, 1)
    sig = load_sigma()
    surv = {}
    for b, pts in blocks.items():
        pts = [p for p in pts if p in us1]
        if not pts:
            continue
        lead = min(us1[p] for p in pts)
        cls = tclass(sweep.parse_point(pts[0])[1])
        s = [p for p in pts if us1[p] <= lead * (1 + Z * sig.get((cls, dur_bin(us1[p])), SIGMA_DEFAULT) + DELTA)]
        if len(s) > 1:
            surv[b] = s
    time_blocks(dict(surv), stage, 2, m)
    us2 = round_us(stage, 2)
    update_noise(us1, us2, surv)
    val, _ = relative(blocks, {1: us1, 2: us2})
    sig = load_sigma()
    close = {}
    for b, s in surv.items():
        s = sorted((p for p in s if p in val), key=val.get)
        if len(s) > 1:
            cls = tclass(sweep.parse_point(s[0])[1])
            lim = 1 + sig.get((cls, dur_bin(us1.get(s[0], 100))), SIGMA_DEFAULT)
            if val[s[0]] * (1 + DELTA) < val[s[1]] <= val[s[0]] * lim:
                close[b] = s[:2]
    time_blocks(dict(close), stage, 3, m)
    us3 = round_us(stage, 3)
    val, _ = relative(blocks, {1: us1, 2: us2, 3: us3})
    log(f"{stage}: {len(blocks)} blocks, {sum(map(len, blocks.values()))} points; round 2 {len(surv)} blocks "
        f"{sum(map(len, surv.values()))} points; round 3 {len(close)} blocks")
    return val, surv


def update_noise(us1, us2, surv):
    """This run's noise model from the survivors timed in rounds 1 and 2 of every stage so far (within-block)."""
    path = os.path.join(OUT, "noise_pairs.json")
    pairs = json.load(open(path)) if os.path.exists(path) else {}
    for b, s in surv.items():
        s = [p for p in s if p in us1 and p in us2]
        if len(s) < 2:
            continue
        g1 = math.exp(statistics.mean(math.log(us1[p]) for p in s))
        g2 = math.exp(statistics.mean(math.log(us2[p]) for p in s))
        for p in s:
            key = f"{tclass(sweep.parse_point(p)[1])} {dur_bin(us1[p])}"
            pairs.setdefault(key, {})[p] = abs((us1[p] / g1) / (us2[p] / g2) - 1)
    json.dump(pairs, open(path, "w"))
    model = {}
    for k, d in pairs.items():
        v = sorted(d.values())
        model[k] = dict(n=len(v), p50=v[len(v) // 2], p90=v[min(len(v) - 1, int(0.9 * len(v)))],
                        p99=v[min(len(v) - 1, int(0.99 * len(v)))])
    json.dump(model, open(os.path.join(OUT, "noise.json"), "w"), indent=1)


# ------------------------------------------------------------------------------------------------ stages


def cells_path(stage):
    return os.path.join(OUT, f"cells_{stage}.json")


def save_cells(stage, cells):
    json.dump(cells, open(cells_path(stage), "w"), indent=0)


def load_cells(stage):
    return json.load(open(cells_path(stage))) if os.path.exists(cells_path(stage)) else None


def stage_blocks(cells):
    return {c["id"]: block_points(c, methods(c["tile"], c["M"], c["N"], c["K"], c["wc"])) for c in cells}


def neighbours(cells):
    """Adjacent pairs of cells (same tile and wave class) along S or k in the cell set's grid."""
    by = collections.defaultdict(dict)
    for c in cells:
        by[(c["tile"], c["wc"])][(c["S"], c["k"])] = c
    out = []
    for g in by.values():
        Ss, ks = sorted({s for s, _ in g}), sorted({k for _, k in g})
        for (s, k), c in g.items():
            i, j = Ss.index(s), ks.index(k)
            if i + 1 < len(Ss) and (Ss[i + 1], k) in g:
                out.append((c, g[(Ss[i + 1], k)], "S"))
            if j + 1 < len(ks) and (s, ks[j + 1]) in g:
                out.append((c, g[(s, ks[j + 1])], "k"))
    return out


def cell_val(c, val):
    """{kind: rel over the cell's best} of a cell."""
    pts = {sweep.parse_point(p)[0]: v for p, v in val.items() if p.split(":", 2)[2] == f"{c['M']}x{c['N']}x{c['K']}"
           and sweep.parse_point(p)[1] == c["tile"]}
    if not pts:
        return {}
    b = min(pts.values())
    return {k: v / b for k, v in pts.items()}


def refine(cells, val, stage, sig=SIGMA_DEFAULT, budget=None):
    """New cells between neighbours whose picks cost each other more than sigma."""
    vals = {c["id"]: cell_val(c, val) for c in cells}
    picks = {c["id"]: pick_kind(vals[c["id"]]) for c in cells}
    have = {(c["tile"], c["wc"], c["S"], c["k"]) for c in cells}
    new = {}
    for a, b, axis in neighbours(cells):
        pa, pb = picks[a["id"]], picks[b["id"]]
        if not pa or not pb or pa == pb:
            continue
        cross = max(vals[a["id"]].get(pb, 9.0), vals[b["id"]].get(pa, 9.0))
        if cross <= 1 + sig:
            continue
        if axis == "S":
            # Geometric midpoint on the doubling part of the S grid, arithmetic on its linear part above P / 2.
            S = (a["S"] + b["S"]) // 2 if b["S"] < 2 * a["S"] else round(math.sqrt(a["S"] * b["S"]))
            k = a["k"]
        else:
            S, k = a["S"], 2 * round(math.sqrt(a["k"] * b["k"]) / 2)
        if (a["tile"], a["wc"], S, k) in have or (S, k) in ((a["S"], a["k"]), (b["S"], b["k"])):
            continue
        sh = cell_shape(a["tile"], a["wc"], S, k, S + k)
        if sh:
            c = make_cell(a["tile"], a["wc"], S, k, sh, stage)
            new[c["id"]] = (cross, c)
    out = [c for _, c in sorted(new.values(), key=lambda x: -x[0])]
    return out[:budget] if budget else out


def pick_kind(v):
    if not v:
        return None
    best = min(v.values())
    return min((k for k in v if v[k] <= best * (1 + DELTA)), key=lambda k: (SIMPLE.index(k) if k in SIMPLE else 99, v[k]))


def n_blocks(cells, val):
    """Stage N: per cell whose best one-tile point is rs at a coarse n, the explicit n between its coarse neighbours,
    with the cell's leader and runner-up."""
    out = {}
    for c in cells:
        v = cell_val(c, val)
        if not v:
            continue
        order = sorted(v, key=v.get)
        lead = order[0]
        fam, n = (lead[:2], lead[2:]) if lead[:2] in ("rs",) and lead[2:].isdigit() else (None, None)
        if not fam:
            continue
        n = int(n)
        i = COARSE_N.index(n) if n in COARSE_N else None
        if i is None:
            continue
        lo = COARSE_N[i - 1] if i > 0 else 1
        hi = COARSE_N[i + 1] if i + 1 < len(COARSE_N) else 10 ** 9
        extra = [f"rs{x}" for x in REFINE_N if lo < x < hi and sweep.valid(f"rs{x}", c["tile"], c["M"], c["N"], c["K"])]
        if extra:
            out[c["id"] + "|n"] = block_points(c, order[:2] + extra)
    return out


def run(args):
    os.environ.setdefault("TRITON_CACHE_DIR", TRITON_CACHE)
    assert os.environ["TRITON_CACHE_DIR"] == TRITON_CACHE, "use the split sweep's own Triton cache"
    m = manifest()
    check_frozen(m)
    cells = load_cells("A") or coarse_cells()
    if args.max_cells:
        cells = random.Random(SEED).sample(cells, args.max_cells)
    save_cells("A", cells)
    val, _ = race("A", stage_blocks(cells), m, args.jobs)
    allcells = list(cells)
    for st in ("B1", "B2"):
        new = load_cells(st)
        if new is None:
            new = refine(allcells, val, st, budget=args.refine_budget)
            save_cells(st, new)
        if new:
            v2, _ = race(st, stage_blocks(new), m, args.jobs)
            val.update(v2)
        allcells += new
        log(f"{st}: {len(new)} new cells")
    nb = n_blocks(allcells, val)
    if nb:
        vn, _ = race("N", nb, m, args.jobs)
        json.dump(vn, open(os.path.join(OUT, "val_N.json"), "w"))
    log("all stages done")


# ------------------------------------------------------------------------------------------------ plan, report


def plan(args):
    cells = coarse_cells()
    blocks = stage_blocks(cells)
    by_tile = collections.Counter(c["tile"] for c in cells)
    pts = sum(map(len, blocks.values()))
    est = sum(sweep.point_cost(p) for v in blocks.values() for p in v)
    print(f"stage A: {len(cells)} cells, {pts} points; one pass about {est / 3600:.1f} h (sweep.point_cost)")
    for t in QUAD_TILES + SMALL_TILES:
        tp = sum(len(blocks[c["id"]]) for c in cells if c["tile"] == t)
        print(f"  {t}: {by_tile[t]} cells, {tp} points, P = {geom(t)[3]}, S grid {s_grid(geom(t)[3])}")
    print(f"racing (Phase 0 replay: about 1.35 passes) about {1.35 * est / 3600:.1f} h; refinement and n stages add "
          f"about 30%")


def report(args):
    """tables.json and tables.md: per tile and wave class, the S x k matrix of picks, cap's and dp's loss, and per
    method its wins, cells within sigma and geomean loss if used in every cell where it is valid."""
    stages = [s for s in ("A", "B1", "B2") if load_cells(s)]
    cells = [c for s in stages for c in load_cells(s)]
    val = {}
    for s in stages:
        blocks = stage_blocks(load_cells(s))
        v, _ = relative(blocks, {r: round_us(s, r) for r in (1, 2, 3)})
        val.update(v)
    tables, L = {}, ["# Split tables (results/split)", ""]
    for tile in QUAD_TILES + SMALL_TILES:
        for wc in WAVE_CLASSES:
            cs = [c for c in cells if c["tile"] == tile and c["wc"] == wc]
            if not cs:
                continue
            Ss, ks = sorted({c["S"] for c in cs}), sorted({c["k"] for c in cs})
            grid = {(c["S"], c["k"]): cell_val(c, val) for c in cs}
            ent = []
            for (S, k), v in sorted(grid.items()):
                if not v:
                    continue
                order = sorted(v, key=v.get)
                pk = pick_kind(v)
                ent.append(dict(S=S, k=k, pick=pk, runner_up=next((o for o in order if o != pk), None),
                                margin=round(min((v[o] for o in order if o != pk), default=9) / v[pk], 4),
                                cap=round(v.get("cap", float("nan")), 4), dp=round(v.get("dp", float("nan")), 4),
                                vals={kk: round(x, 4) for kk, x in v.items()}))
            tables[f"{tile} w{wc}"] = ent
            L += [f"## {tile}, wave class {wc}", "", "Picks (rows S, columns k-steps); in brackets cap's time over the "
                  "cell's best.", "", "| S \\ k | " + " | ".join(map(str, ks)) + " |", "|---" * (len(ks) + 1) + "|"]
            for S in Ss:
                row = []
                for k in ks:
                    v = grid.get((S, k))
                    row.append(f"{pick_kind(v)} ({v.get('cap', float('nan')):.2f})" if v else "")
                L.append(f"| {S} | " + " | ".join(row) + " |")
            stats = collections.defaultdict(lambda: [0, 0, []])
            for e in ent:
                for kk, x in e["vals"].items():
                    stats[kk][2].append(x)
                    stats[kk][1] += x <= 1 + SIGMA_DEFAULT
                stats[e["pick"]][0] += 1
            L += ["", "| method | wins | within sigma | cells timed | geomean loss | worst |", "|---|---|---|---|---|---|"]
            for kk, (w, ws, xs) in sorted(stats.items(), key=lambda kv: (-kv[1][0], -kv[1][1])):
                g = math.exp(statistics.mean(math.log(x) for x in xs))
                L.append(f"| {kk} | {w} | {ws} | {len(xs)} | {g:.3f} | {max(xs):.3f} |")
            L.append("")
    json.dump(tables, open(os.path.join(OUT, "tables.json"), "w"), indent=0)
    open(os.path.join(OUT, "tables.md"), "w").write("\n".join(L) + "\n")
    print(f"{sum(map(len, tables.values()))} cells in {OUT}/tables.json and tables.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["plan", "run", "report"])
    ap.add_argument("--jobs", type=int, default=32, help="warm compile processes")
    ap.add_argument("--refine-budget", type=int, default=600, help="most new cells per refinement stage")
    ap.add_argument("--max-cells", type=int, default=0, help="a seeded sample of stage A's cells (smoke tests, with "
                    "SPLIT_OUT elsewhere)")
    args = ap.parse_args()
    {"plan": plan, "run": run, "report": report}[args.mode](args)


if __name__ == "__main__":
    main()
