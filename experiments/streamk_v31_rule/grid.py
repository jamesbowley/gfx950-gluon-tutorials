"""Shapes, tiles and configurations of the v31 rule sweep (CPU only).

A point is ``KIND:BMxBNxBK:MxNxK``. Kinds:
- ``dp``: data-parallel;
- ``ow{n}`` / ``rs{n}``: one_tile with the owner / reduce-scatter ending on n
  programs per tail tile (write-through exchange);
- ``l2rs{n}``: one_tile reduce-scatter with the XCD-local L2 exchange;
- ``2t``, ``sp``: two_tile and spread.

Shapes are built in stream-K coordinates: for a tile, T = W P + S output tiles
(W full waves, S tail tiles) and k K-steps per tile. A (shape, tile) pair is kept
only where a tuner could pick the tile: the shape is in the domain (in_domain) and
the tile's cost under AITER's gfx950 gluon tile model (pick_gemm_tile's
TILE_TFLOPS and TAIL_WAVE_COST) is within TILE_SLACK of the best tile's.
"""

import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A16 = os.path.join(HERE, "..", "..", "kernels", "gemm", "intra_wave", "a16w16")
sys.path.insert(0, os.path.abspath(A16))

P = 256
NUM_XCDS = 8
# AITER's configs/gfx950/gluon/gemm/gemm_a16w16/DEFAULT.json (llirSched_integration).
MODEL_TILES = ((256, 256), (256, 128), (128, 256), (128, 128), (256, 64), (64, 256),
               (128, 64), (64, 128), (64, 64))
MODEL_TFLOPS = (1583, 1386, 1388, 1162, 1040, 1009, 877, 871, 603)
TAIL_WAVE_COST = 0.5
TILE_SLACK = 1.25
MIN_INTENSITY = 128

# The sweep's tile classes (BLOCK_K as the v29 auto rule uses them) and spot checks of
# the transposed orientation.
TILE_CLASSES = ("256x256x64", "256x128x64", "128x128x64", "128x128x128", "128x64x128", "64x64x128")
SPOT_TILES = ("128x256x64", "64x128x128")
OWNER_N = (2, 3, 4, 6, 8, 12, 16)
RS_N = (2, 4, 6, 8, 12, 16, 32, 64)
COARSE_S = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)
COARSE_W = (0, 1, 2, 3)
COARSE_KSTEPS = (8, 16, 24, 32, 48, 64, 96, 128, 256)
SPOT_SHARE = 0.1
SEED = 3101


def cdiv(a, b):
    return -(-a // b)


def parse_tile(tile):
    bm, bn, bk = (int(x) for x in tile.split("x"))
    return bm, bn, bk


def parse_point(p):
    kind, tile, shape = p.split(":")[:3]
    M, N, K = (int(x) for x in shape.split("x"))
    return kind, tile, M, N, K


def point_name(kind, tile, M, N, K):
    return f"{kind}:{tile}:{M}x{N}x{K}"


def tile_cost(bm, bn, M, N, tail=TAIL_WAVE_COST):
    """pick_gemm_tile's cost of a tile: waves of P workgroups, a partial last wave
    costing `tail` plus its fill, each wave the tile's work over its TFLOPS."""
    tflops = dict(zip(MODEL_TILES, MODEL_TFLOPS))[(bm, bn)]
    full, rem = divmod(cdiv(M, bm) * cdiv(N, bn), P)
    waves = full + (0 if rem == 0 else tail + (1 - tail) * rem / P)
    return waves * bm * bn / tflops


def model_costs(M, N, tail=TAIL_WAVE_COST):
    return {t: tile_cost(*t, M, N, tail) for t in MODEL_TILES}


def pick_gemm_tile(M, N):
    """AITER's pick: the lowest cost, ties within 1% to the earlier (larger) tile."""
    costs = model_costs(M, N)
    best = min(costs.values())
    return next(t for t in MODEL_TILES if costs[t] <= 1.01 * best)


def plausible_tile(bm, bn, M, N, slack=TILE_SLACK):
    """Within slack of the best tile under pick_gemm_tile's model, or under the same
    model with an ideal tail (the last wave costs only its fill, as a perfect
    stream-K split would make it): stream-K competes with the tile choice."""
    for tail in (TAIL_WAVE_COST, 0.0):
        costs = model_costs(M, N, tail)
        if costs[(bm, bn)] <= slack * min(costs.values()):
            return True
    return False


def intensity(M, N, K):
    return M * N * K / (M * K + N * K + M * N)


def in_domain(M, N, K):
    """Shapes for tiles with sides of at least 64: 64x64 tiles fill at least one
    wave (else a smaller tile or a split-K kernel is the tool), the GEMM is near or
    above the compute ridge, and K is long enough to split at all."""
    return cdiv(M, 64) * cdiv(N, 64) >= P and intensity(M, N, K) >= MIN_INTENSITY and K >= 512


def factor_shape(T, bm, bn, max_aspect=32):
    """(tiles_m, tiles_n) with tiles_m tiles_n = T, both at least 2 (T <= 3: any),
    closest to square in elements, aspect at most max_aspect; None if none."""
    best = None
    for tm in range(1, T + 1):
        if T % tm:
            continue
        tn = T // tm
        if T > 3 and (tm < 2 or tn < 2):
            continue
        aspect = max(tm * bm, tn * bn) / min(tm * bm, tn * bn)
        if aspect > max_aspect:
            continue
        if best is None or aspect < best[0]:
            best = (aspect, tm, tn)
    return None if best is None else best[1:]


def realize(W, S, bm, bn):
    """A shape with W full waves and about S tail tiles: T nudged by up to 2 when T
    has no usable factorization. Returns (M, N, S_actual) or None."""
    for d in (0, 1, -1, 2, -2):
        s = S + d
        if not 1 <= s < P:
            continue
        f = factor_shape(W * P + s, bm, bn)
        if f:
            return f[0] * bm, f[1] * bn, s
    return None


def shape_info(tile, M, N, K):
    bm, bn, bk = parse_tile(tile)
    T = cdiv(M, bm) * cdiv(N, bn)
    k_pad = cdiv(K, 2 * bk) * 2 * bk
    return dict(T=T, S=T % P, W=T // P, k=k_pad // bk, pairs=k_pad // (2 * bk))


def config_for(kind, tile):
    bm, bn, bk = parse_tile(tile)
    c = dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE_M=4, STREAMK_POLICY="dp",
             STREAMK_FIXUP="owner", SPLIT_N=None, STREAMK_EXCHANGE="wt")
    if kind == "dp":
        return c
    if kind == "def":
        return dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)
    if kind == "2t" or kind.startswith("2tg"):
        return dict(c, STREAMK_POLICY="two_tile")
    if kind == "sp":
        return dict(c, STREAMK_POLICY="spread")
    for prefix, fixup, exchange in (("l2rs", "rs", "l2"), ("rs", "rs", "wt"), ("ow", "owner", "wt")):
        if kind.startswith(prefix):
            return dict(c, STREAMK_POLICY="one_tile", STREAMK_FIXUP=fixup,
                        SPLIT_N=int(kind[len(prefix):]), STREAMK_EXCHANGE=exchange)
    raise ValueError(kind)


def streamk_grid(kind):
    """G of a ``2tg{G}`` kind (two_tile on G programs), else None."""
    return int(kind[3:]) if kind.startswith("2tg") else None


def valid(kind, tile, M, N, K):
    """The point runs. Uses the offline resolver (test_selection checks it against
    the kernel's launch constants), so this process never opens the GPU."""
    from v31_streamk_aiter.selection import DeviceConfig, try_resolve_config

    G = streamk_grid(kind)
    if G is not None:
        T, pairs = shape_info(tile, M, N, K)["T"], shape_info(tile, M, N, K)["pairs"]
        return G % NUM_XCDS == 0 and G <= P and G < T < 2 * G and T * pairs >= G and pairs >= 2
    if kind == "def":
        return True
    device = DeviceConfig("gfx950", P, NUM_XCDS, 160 * 1024)
    if try_resolve_config(M, N, K, config=config_for(kind, tile), device=device) is None:
        return False
    bk = parse_tile(tile)[2]
    return cdiv(K, 2 * bk) * 2 * bk >= 4 * bk


def kinds_for(tile, M, N, K, spot=False):
    """dp, owner and rs at every listed n that runs; with spot, also two_tile, spread
    and l2 at the largest valid n up to 8."""
    kinds = ["dp"] + [f"ow{n}" for n in OWNER_N] + [f"rs{n}" for n in RS_N]
    if spot:
        kinds += ["2t", "sp"] + [f"l2rs{n}" for n in (8, 6, 4, 2)]
    out = [k for k in kinds if valid(k, tile, M, N, K)]
    l2 = [k for k in out if k.startswith("l2rs")]
    return [k for k in out if not k.startswith("l2rs")] + l2[:1]


def coarse_shapes():
    """(tile, M, N, K, W, S_target) of the coarse grid, filtered to plausible pairs."""
    out = []
    for tile in TILE_CLASSES:
        bm, bn, bk = parse_tile(tile)
        for W in COARSE_W:
            for S in COARSE_S:
                r = realize(W, S, bm, bn)
                if r is None:
                    continue
                M, N, _ = r
                for k in COARSE_KSTEPS:
                    K = k * bk
                    if in_domain(M, N, K) and plausible_tile(bm, bn, M, N):
                        out.append((tile, M, N, K, W, S))
    return out


def coarse_points():
    rng = random.Random(SEED)
    pts = []
    for tile, M, N, K, W, S in coarse_shapes():
        spot = rng.random() < SPOT_SHARE
        pts += [point_name(k, tile, M, N, K) for k in kinds_for(tile, M, N, K, spot)]
    return pts


AITER_SHAPES_CSV = os.path.join(HERE, "..", "aiter_lixun_shapes", "supported_status_scoped.csv")
HELDOUT_TILES = 3


def model_tile_spec(bm, bn):
    """The BLOCK_K a model tile runs with: 128 for 128x64, 64x128 and 64x64, else 64."""
    bk = 128 if (bm, bn) in ((128, 64), (64, 128), (64, 64)) else 64
    return f"{bm}x{bn}x{bk}"


def heldout_shapes():
    """AITER's bf16 shapes (distinct M, N, K) in the domain that the kernel runs."""
    import csv

    out = []
    for r in csv.DictReader(open(AITER_SHAPES_CSV)):
        M, N, K = int(r["M"]), int(r["N"]), int(r["K"])
        if N % 8 or K % 16 or not in_domain(M, N, K):
            continue
        out.append((M, N, K))
    return sorted(set(out))


def heldout_points():
    """Every valid dp / owner / rs point at each held-out shape's
    HELDOUT_TILES cheapest tiles under pick_gemm_tile's model (its pick first)."""
    pts = []
    for M, N, K in heldout_shapes():
        costs = model_costs(M, N)
        pick = pick_gemm_tile(M, N)
        tiles = [pick] + sorted((t for t in MODEL_TILES if t != pick), key=costs.get)[: HELDOUT_TILES - 1]
        for bm, bn in tiles:
            tile = model_tile_spec(bm, bn)
            pts += [point_name(k, tile, M, N, K) for k in kinds_for(tile, M, N, K)]
    return pts


GAP_TILES = ("256x256x64", "256x128x64", "128x128x64")
GAP_S = (12, 18, 22, 26, 34, 36, 38, 40, 44, 50, 56, 72, 80)
GAP_W = (0, 1, 2)
GAP_KSTEPS = (8, 12, 16, 20, 24, 32, 48, 64, 128, 256)


def gap_points():
    """The S (P // S = 3-21, most 4-7) and k values the grid skipped, every valid dp,
    owner and rs point, at plausible pairs: a test of the rule between the grid's
    sample points."""
    pts = []
    for tile in GAP_TILES:
        bm, bn, bk = parse_tile(tile)
        for W in GAP_W:
            for S in GAP_S:
                r = realize(W, S, bm, bn)
                if r is None:
                    continue
                M, N, _ = r
                for k in GAP_KSTEPS:
                    K = k * bk
                    if in_domain(M, N, K) and plausible_tile(bm, bn, M, N):
                        pts += [point_name(kd, tile, M, N, K) for kd in kinds_for(tile, M, N, K)]
    return list(dict.fromkeys(pts))


TILESWEEP_SHAPES = 300
TILESWEEP_TILES = 4
TILESWEEP_SEED = 3103


def tilesweep_shapes():
    """Seeded shapes for fitting the tile pick: M and N log-uniform in 512-6144
    (multiples of 64), K in 1024-16384, in the domain and not held-out shapes."""
    rng = random.Random(TILESWEEP_SEED)
    held = set(heldout_shapes())
    out = set()
    while len(out) < TILESWEEP_SHAPES:
        M = 64 * round(math.exp(rng.uniform(math.log(512), math.log(6144))) / 64)
        N = 64 * round(math.exp(rng.uniform(math.log(512), math.log(6144))) / 64)
        K = rng.choice((1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384))
        if in_domain(M, N, K) and (M, N, K) not in held:
            out.add((M, N, K))
    return sorted(out)


def tilesweep_points():
    """Each shape's TILESWEEP_TILES cheapest tiles under the dp model or the ideal-tail
    model (whichever ranks the tile higher), at every pruned candidate of the rule."""
    import json

    import candidates

    rule = json.load(open(os.path.join(HERE, "results", "rule_safe.json")))
    pts = []
    for M, N, K in tilesweep_shapes():
        rank = {}
        for tail in (TAIL_WAVE_COST, 0.0):
            costs = model_costs(M, N, tail)
            for i, t in enumerate(sorted(MODEL_TILES, key=lambda t: (costs[t], MODEL_TILES.index(t)))):
                rank[t] = min(rank.get(t, i), i)
        for bm, bn in sorted(MODEL_TILES, key=lambda t: (rank[t], MODEL_TILES.index(t)))[:TILESWEEP_TILES]:
            tile = model_tile_spec(bm, bn)
            t3 = tuple(int(x) for x in tile.split("x"))
            for cfg in candidates.streamk_candidates(M, N, K, t3, P, rule):
                kind = "dp" if cfg["STREAMK_POLICY"] == "dp" else (
                    ("rs" if cfg["STREAMK_FIXUP"] == "rs" else "ow") + str(cfg["SPLIT_N"]))
                if valid(kind, tile, M, N, K):
                    pts.append(point_name(kind, tile, M, N, K))
    return list(dict.fromkeys(pts))


# Every tile spec the kernel runs: BLOCK_K 64 for every tile, and 128 where it fits LDS
# (both sides at most 128).
ALL_SPECS = ("256x256x64", "256x128x64", "128x256x64", "128x128x64", "128x128x128", "256x64x64",
             "64x256x64", "128x64x64", "128x64x128", "64x128x64", "64x128x128", "64x64x64", "64x64x128")
TIMED_CSVS = ("grid", "tilesweep", "gap", "heldout")


def timed_points():
    import csv

    out = set()
    for name in TIMED_CSVS:
        path = os.path.join(HERE, "results", f"{name}_r1.csv")
        if os.path.exists(path):
            out.update(r["point"] for r in csv.DictReader(open(path)))
    return out


def joint_points(shapes):
    """Every valid dp / owner / rs point at every spec in ALL_SPECS, less those already
    timed (the analysis reads both)."""
    done = timed_points()
    pts = []
    for M, N, K in shapes:
        for tile in ALL_SPECS:
            pts += [p for p in (point_name(k, tile, M, N, K) for k in kinds_for(tile, M, N, K))
                    if p not in done]
    return list(dict.fromkeys(pts))


# FORMULA: every spec in the formula's coordinates: W full waves, m = P // S programs per
# tail tile, k K-steps per tile; FORMULA_SHAPES shapes per cell (different factorizations,
# S nudged within the same m), the domain filter only (the formula is "given the tile").
# Cells the earlier sweeps already hold FORMULA_SHAPES shapes in are skipped.
FORMULA_M = (2, 3, 4, 6, 10, 20, 40, 100)
FORMULA_W = (0, 1, 2, 3)
FORMULA_KSTEPS = (8, 12, 16, 24, 48, 96, 256)
FORMULA_SHAPES = 3
FORMULA_OWNER_N = (2, 3, 4)
FORMULA_RS_N = (2, 4, 8, 16, 32, 64)
COVER_M = ((2, 2), (3, 3), (4, 4), (5, 7), (8, 15), (16, 31), (32, 63), (64, 256))
COVER_K = ((8, 8), (12, 16), (20, 24), (32, 48), (64, 96), (128, 256))


def cover_cell(spec, M, N, K):
    """(spec, regime, m bin, k bin) of a pair, None without a tail."""
    g = shape_info(spec, M, N, K)
    if not g["S"]:
        return None
    m = P // g["S"]
    mi = next((i for i, (lo, hi) in enumerate(COVER_M) if lo <= m <= hi), None)
    ki = next((i for i, (lo, hi) in enumerate(COVER_K) if lo <= g["k"] <= hi), None)
    if mi is None or ki is None:
        return None
    return (spec, "below" if g["T"] < P else "after", mi, ki)


def timed_pairs(names=TIMED_CSVS + ("joint_train",)):
    import csv

    out = set()
    for name in names:
        path = os.path.join(HERE, "results", f"{name}_r1.csv")
        if os.path.exists(path):
            out.update((r["tile"], int(r["M"]), int(r["N"]), int(r["K"])) for r in csv.DictReader(open(path)))
    return out


def _factorizations(T, bm, bn, max_aspect=16):
    out = []
    for tm in range(2, T // 2 + 1):
        if T % tm == 0:
            tn = T // tm
            aspect = max(tm * bm, tn * bn) / min(tm * bm, tn * bn)
            if aspect <= max_aspect:
                out.append((aspect, tm, tn))
    return sorted(out)


def formula_shapes(spec, W, m):
    """Up to FORMULA_SHAPES distinct (M, N) with W full waves and P // S == m."""
    bm, bn, _ = parse_tile(spec)
    s0 = max(1, round(P / m))
    shapes = []
    for d in range(0, 16):
        for s in ((s0,) if d == 0 else (s0 + d, s0 - d)):
            if not 1 <= s < P or P // s != P // s0:
                continue
            for _, tm, tn in _factorizations(W * P + s, bm, bn)[:2]:
                if (tm * bm, tn * bn) not in shapes:
                    shapes.append((tm * bm, tn * bn))
        if len(shapes) >= FORMULA_SHAPES:
            break
    return shapes[:FORMULA_SHAPES]


def formula_pairs():
    have = collections_counter(cover_cell(*p) for p in timed_pairs())
    pairs = []
    for spec in ALL_SPECS:
        bk = parse_tile(spec)[2]
        for W in FORMULA_W:
            for m in FORMULA_M:
                for M, N in formula_shapes(spec, W, m):
                    for k in FORMULA_KSTEPS:
                        K = k * bk
                        c = cover_cell(spec, M, N, K)
                        if c is None or not in_domain(M, N, K) or have[c] >= FORMULA_SHAPES:
                            continue
                        pairs.append((spec, M, N, K))
    return list(dict.fromkeys(pairs))


def collections_counter(items):
    import collections

    return collections.Counter(x for x in items if x is not None)


def formula_points():
    done = timed_points()
    pts = []
    for spec, M, N, K in formula_pairs():
        m = P // shape_info(spec, M, N, K)["S"]
        kinds = ["dp"] + [f"ow{n}" for n in FORMULA_OWNER_N if m <= 8] + [f"rs{n}" for n in FORMULA_RS_N]
        valid_kinds = [k for k in kinds if valid(k, spec, M, N, K)]
        if len(valid_kinds) > 1:
            pts += [point_name(k, spec, M, N, K) for k in valid_kinds]
    return [p for p in dict.fromkeys(pts) if p not in done]


def coverage_report(extra_pairs=()):
    """Per spec, cells (of len(COVER_M) x len(COVER_K) per regime) with at least
    FORMULA_SHAPES shapes, in the timed data plus extra_pairs."""
    cov = collections_counter(cover_cell(*p) for p in set(timed_pairs()) | set(extra_pairs))
    lines = []
    for spec in ALL_SPECS:
        row = []
        for regime in ("after", "below"):
            cells = {c: v for c, v in cov.items() if c[0] == spec and c[1] == regime}
            row.append(f"{regime} {sum(v >= FORMULA_SHAPES for v in cells.values()):2d}/{len(cells):2d}")
        lines.append(f"{spec:12s} " + "   ".join(row))
    return "\n".join(lines)


def est_us(tile, M, N, K):
    bm, bn, _ = parse_tile(tile)
    tf = dict(zip(MODEL_TILES, MODEL_TFLOPS)).get((bm, bn), 600)
    return 2 * M * N * K / (tf * 1e12) * 1e6 + 3


def shape_iters(tile, M, N, K, graph_len=100):
    t = est_us(tile, M, N, K)
    return 1000 if t <= 250 else max(2 * graph_len, graph_len * math.ceil(2.5e5 / t / graph_len))


if __name__ == "__main__":
    shapes = coarse_shapes()
    pts = coarse_points()
    by_tile = {}
    for tile, *_ in shapes:
        by_tile[tile] = by_tile.get(tile, 0) + 1
    print(f"{len(shapes)} shapes, {len(pts)} points; per tile {by_tile}")
    est = sum(shape_iters(t, M, N, K) * est_us(t, M, N, K) for t, M, N, K in
              (parse_point(p)[1:] for p in pts)) / 1e6
    print(f"kernel time in timed launches: {est / 3600:.2f} h (one round)")
