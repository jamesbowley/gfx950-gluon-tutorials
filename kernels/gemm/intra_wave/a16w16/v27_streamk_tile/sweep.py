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
"""Back-to-back timing of v27 against v9_tile's data-parallel kernel, per tile.

    source v27_streamk_tile/env.sh
    python v27_streamk_tile/sweep.py --points WQ4096 --out v27_streamk_tile/results/wq4096.csv
    python v27_streamk_tile/sweep.py --points MB --out v27_streamk_tile/results/mb.csv
    python v27_streamk_tile/sweep.py warm --points PRELIM --jobs 48      # compile only, no launches
    python v27_streamk_tile/sweep.py --points PRELIM --shard 3/8 --out v27_streamk_tile/results/prelim_3.csv

A point is "KIND:TILE:MxNxK", with ":r" appended for a noise repeat:
- dp: v9_tile at TILE (data-parallel, one program per tile; a small tile with one wave per SIMD, as v27 runs it);
- pdp: v27's persistent data-parallel loop at TILE (STREAMK_POLICY=dp, which never goes to v9_tile);
- pdp{P} (P = 512, 768, 1024): the same on P programs (STREAMK_PERSISTENT_PROGRAMS), quadrant tiles whose build fits
  ceil(P / 256) workgroups per CU (PDP_MAX_PROGRAMS);
- cap: v27 at TILE as deployed (STREAMK_POLICY=capped, the default; its data-parallel choices of one
  wave or less go to v9_tile, larger ones run its persistent loop);
- rs{n}, ow{n}, az{n}: v27's one-tile split with the reduce-scatter (pow2 readers), owner or atomic_iz
  ending on n programs per leftover tile (STREAMK_NUM_PROGRAMS = S n; n = all: every program, so
  min(programs // S, pairs) per tile, the first programs % S tiles one more); a suffix b or p on rs / ow gives
  bf16 or bf16-pair partials (STREAMK_PARTIALS; default fp32);
- 2t: two-tile (one full wave moved into stream-K, v19), when there is a wave to borrow; sp: spread
  (two-tile's end-to-end reversed split on the leftover tiles). Both end with the owner's serial collection;
- all: v27's one-tile split over every program with the auto rule's ending (v9_tile §13's sk256);
- any v27 kind with -full appended (cap-full, all-full, ...): the same on the full grid (STREAMK_COMPACT_GRID=0).
S is the leftover tile count (tiles % programs, on the kernel's dimensions: a swapped small tile runs
C^T). valid() drops a split that cannot run or repeats another: no leftover tiles, n > min(programs // S,
pairs), an explicit n that is what `all` runs, fewer than 2 programs per tile, 2t without a full wave, sp
with fewer pairs than programs.

Each point is checked against torch (the atomic ending and bf16 partials, which round every partial, by
check_streamk.py's per-tile mean error), then replayed from a HIP graph of 100 launches over 2 GB of
rotating inputs (cold caches, back to back) under rocprofv3 --kernel-trace: the median of the last
100 dispatches. Besides the timing, a row records what ran: the leftover tiles S, waves and k-steps of
the tile; n (n_hi for the first tiles when uneven; programs per tile on average for 2t and sp), the
ending (dp, rs, owner, atomic_iz, e2e), programs (with work in the stream-K phase, or the last wave for
data-parallel), pairs_per_prog (pairs of k-steps per program in that phase) and busy_cus (an estimate,
min(programs, 256): programs are spread evenly over the XCDs, and the first 32 of an XCD land on
distinct CUs). Presets (POINTS):
- WQ4096, WQ8192: v9_tile WORKLOG §13's wave-quantization sweep (M = 4096, N = 256 .. 16384 in steps of
  256) at the quadrant tiles WQ_TILES, dp and cap at every N, all at §13's 16 stream-K N.
- MB: §14.7's memory-bound grid (M = 1, 4, 16, 32 = BM, N = 4096 / 8192 / 16384, BN = 16 .. 256, BK = 128,
  K = 8192), dp, cap and all, each tile with the fewest waves that pin (pinned_spec); TB/s counts A, B
  and C once.
- PRELIM: the preliminary factorial sweep (WORKLOG §11; az and the b / p partials are opt-in and left out): every
  valid PRELIM_KINDS point at the quadrant
  tiles on the compute-bound shapes PRELIM_CB, and at all 7 tiles on the memory-bound shapes PRELIM_MB,
  plus a seeded 5% of them again as noise repeats, which --shard puts on another shard (GPU), after its
  own points.
- FULL: every FULL_KINDS point (n = 2-8, 12, 16, all; pdp) at the quadrant tiles on FULL_CB (PRELIM's compute-bound
  shapes) and at every tile v27 accepts that passed its stress check (candidate_tiles, stress_checked) on FULL_MB
  (M = 1-128, N = 512-16384, K = 4096, 8192), plus a seeded 2% as noise repeats, sharded as PRELIM.
- COMPACT: the compact launch's A/B (compact_sets: MB's cap and all, WQ's points with fewer tiles than
  programs, COMPACT_AITER seeded AITER shapes below one wave at 256x256 and 128x128x128), each point as
  KIND-full, KIND, KIND-full:r, KIND:r back to back, whole groups per shard; compact_report.py reports it.
- REFIT: the re-fit sweep (refit.py runs it in rounds): every valid REFIT_KINDS point at the REFIT_TILES that passed
  their stress check on the current kernel (refit_checked), quadrant tiles on refit_shapes()'s compute-bound shapes and
  all of them on its memory-bound ones.
- REFIT2: the same on refit2_shapes() (tall-skinny, large and wide-short compute-bound shapes, a denser memory-bound
  grid up to M = 448, and the structured shapes at exact tile counts, each at its target's structured_tiles), less
  REFIT2_MB_PRUNE, small tiles also needing their ragged-M check (raggedm_checked).

A timing run (timing mode, and refit.py's shards) holds the GPU's check lock (/tmp/v27_check_gpu<N>.lock, which
check_streamk.py and check_grid.py take for their launches) for each chunk of points, and looks for other processes on
the GPU (rocm-smi) before the chunk and again after it; a chunk with one after it is dropped and timed again. With
--frozen MANIFEST it also stops when a file hashed in the manifest (FROZEN_FILES) has changed. Each row records the
GPU, the round (refit.py), the chunk, the point's position in it, the time it finished and the launches timed (iters;
--iters 0: shape_iters, fewer for long kernels).
"""

import argparse
import collections
import contextlib
import csv
import fcntl
import functools
import hashlib
import json
import math
import os
import queue
import random
import re
import signal
import statistics as S
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
A16 = os.path.dirname(HERE)
sys.path.insert(0, A16)
from v9_tile import sweep as base  # noqa: E402

TMP = os.path.join(HERE, "tmp")
GRAPH_LEN = 100
ROTATING_MB = 2048
WQ_TILES = ("256x256", "256x128", "128x256", "128x128x128")
WQ_N = range(256, 16384 + 1, 256)
WQ_SK_N = [w * 4096 + d for w in range(4) for d in (256, 512, 1024, 2048)]
MB_M = (1, 4, 16, 32)
MB_N = (4096, 8192, 16384)
MB_BN = (16, 32, 64, 128, 256)
SMALL_TILES = ("16x256x128", "16x64x128", "32x128x128")
# The atomic ending and 16-bit partials are opt-in (selected explicitly, never by a rule), so the presets leave them
# out. The recorded PRELIM run (WORKLOG §11) also timed azall; prelim_report.py drops it.
PRELIM_KINDS = ("dp", "cap", "rs2", "rs4", "rs8", "rs16", "rsall", "ow2", "ow4", "owall", "2t", "sp")
PRELIM_CB = [(m, n, k) for m in (1024, 4096) for n in (256, 512, 1024, 1536, 2048, 3072, 4352, 5120, 6144, 8448)
             for k in (2048, 8192)]
PRELIM_MB = [(m, n, k) for m in (1, 16, 32, 64, 128) for n in (4096, 8192, 16384) for k in (4096, 8192)]
NOISE_SHARE = 0.05
# FULL: every tile v27 accepts that passed its stress check, every n from 2 to 8 (and 12, 16, all), and the
# persistent data-parallel loop (pdp).
FULL_KINDS = ("dp", "pdp", "cap", "rs2", "rs3", "rs4", "rs5", "rs6", "rs8", "rs12", "rs16", "rsall", "ow2", "ow3",
              "ow4", "ow6", "ow8", "owall", "2t", "sp")
FULL_CB = PRELIM_CB
FULL_MB = [(m, n, k) for m in (1, 16, 32, 64, 128) for n in (512, 1024, 2048, 4096, 8192, 16384) for k in (4096, 8192)]
FULL_NOISE_SHARE = 0.02
# COMPACT: the compact launch's A/B (WORKLOG §13): cap and all, each against its -full twin, on the memory-bound
# grid (MB), the wave-quantization points with fewer tiles than programs (N <= 2048), and COMPACT_AITER seeded
# AITER shapes with less than one wave of 256x256 tiles, at 256x256 and 128x128x128.
COMPACT_KINDS = ("cap", "all")
COMPACT_WQ_N = range(256, 2048 + 1, 256)
COMPACT_AITER = 100
COMPACT_AITER_TILES = ("256x256", "128x128x128")
AITER_SHAPES_CSV = os.path.join(A16, "..", "..", "..", "..", "experiments", "aiter_lixun_shapes",
                                "supported_status_scoped.csv")
# pdp{P}: the persistent data-parallel loop on P programs. PDP_MAX_PROGRAMS is the most each quadrant tile's dp build
# holds resident (tmp/pdp_probe.py: LDS and VGPRs, aligned and with every edge; the others fit 256 only); the host
# checks every build again (check_residency), and refit.py drops a point whose warm compile failed that check.
PDP_PROGRAMS = (512, 768, 1024)
PDP_MAX_PROGRAMS = {"64x64x64": 1024, "64x64x128": 512, "64x128x64": 512, "128x64x64": 512, "128x128x64": 512}
# REFIT (refit.py): FULL's greedy tile covers within noise and within 5% per regime, plus the next two fastest per tile
# class (tmp/refit_tiles.py on results/full.csv), and 64x64x64, the only tile that holds pdp768 and pdp1024.
REFIT_TILES = ("256x256x64", "128x128x128", "128x256x64", "64x128x128", "64x64x128", "128x128x64", "256x128x64",
               "16x16x256", "32x32x128", "1x16x256", "128x64x128", "32x128x128", "128x32x128", "128x64x64",
               "16x16x128", "32x256x128", "64x128x64", "64x256x64", "256x64x64", "16x32x128", "16x64x128",
               "16x256x128", "32x256x64", "16x128x128", "16x128x64", "64x64x64")
REFIT_N = (2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32)
REFIT_KINDS = (("dp", "pdp") + tuple(f"pdp{p}" for p in PDP_PROGRAMS) + ("cap",)
               + tuple(f"{e}{n}" for e in ("rs", "ow") for n in REFIT_N + ("all",)) + ("2t", "sp"))
REFIT_SEED = 2610
# Compute-bound: 256x256 waves x leftover 256x256 tiles (of 256 programs: 1/64-1/16, 1/16-1/4, 1/4-1) cells, each with
# REFIT_CB_PER_CELL shapes spread over the K bins; M = 512-8192, N = 256-16384, both multiples of 256.
REFIT_CB_WAVES = ((0, 0), (1, 3), (4, 7))
REFIT_CB_S = ((4, 16), (16, 64), (64, 256))
REFIT_CB_K = (512, 1024, 2048, 4096, 8192, 16384, 32768)
REFIT_CB_PER_CELL = 22
# Memory-bound: REFIT_MB_PER_M shapes per M, N (256-32768, multiples of 256) and K (bins REFIT_MB_K) Latin-hypercube
# spread in log space.
REFIT_MB_M = (1, 4, 8, 16, 32, 64, 128, 256)
REFIT_MB_K = (1024, 2048, 4096, 8192, 16384)
REFIT_MB_N = (256, 32768)
REFIT_MB_PER_M = 19
# REFIT2 (refit.py --study refit2): the shapes REFIT did not cover, seeded (REFIT2_SEED), none of REFIT's or the
# held-out set's. Compute-bound groups (name, M / 256 range, N / 256 range, 256x256 waves cells, leftover cells, shapes
# per cell), cells that the M and N ranges cannot reach left out; K bins REFIT_CB_K, lowered to keep M N K within
# REFIT2_CB_MNK. Memory-bound: REFIT2_MB_PER_M shapes per M in REFIT2_MB_M, N in log strata of REFIT2_MB_N, K bins
# REFIT2_MB_K lowered to keep N K within REFIT2_MB_NK. A seeded REFIT2_TEST_SHARE of each group (per M for memory-bound)
# is the test split, held back from fitting.
REFIT2_SEED = 2620
REFIT2_CB_GROUPS = (
    ("tall-skinny", (32, 256), (1, 8), ((0, 0), (1, 3), (4, 8)), ((0, 4), (4, 16), (16, 64), (64, 256)), 12),
    ("large", (32, 128), (8, 512), ((1, 3), (4, 15), (16, 63), (64, 256)), ((0, 4), (4, 64), (64, 256)), 5),
    ("wide-short", (2, 8), (64, 512), ((0, 0), (1, 3), (4, 16)), ((0, 4), (4, 64), (64, 256)), 6),
)
REFIT2_CB_MNK = 4e12
REFIT2_MB_M = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 320, 384, 448)
REFIT2_MB_N = (256, 131072)
REFIT2_MB_K = (1024, 2048, 4096, 8192, 16384, 32768)
REFIT2_MB_NK = 2**30
REFIT2_MB_PER_M = 12
REFIT2_TEST_SHARE = 0.2
# Memory-bound tiles left out from this M up: 1x16x256 is at best 1.39x the shape's best tile from M = 32 in REFIT
# (3.65x at M = 256), and its re-reads of B make it the slowest point of a block.
REFIT2_MB_PRUNE = {"1x16x256": 32}
# REFIT2's structured group (refit2_structured, its own seed REFIT2_SEED + 2): for each (BM, BN) on auto_tile's path,
# shapes on which that tile makes exactly a given number of tiles (128 = half a wave, where auto halves the tile;
# 120 and 136 either side; 64 and 192 as controls), {count: (shapes per target, K values)}, M and N multiples of the
# tile in every orientation the ranges allow; timed at the target's tiles, those of half and a quarter its area, and
# 256x256x64 and 128x128x64 (structured_tiles).
REFIT2_STRUCT_TARGETS = ((256, 256), (128, 256), (256, 128), (128, 128), (64, 128))
REFIT2_STRUCT_K = (512, 1024, 2048, 3072, 4096, 4608, 5120, 6144, 8192, 16384)
REFIT2_STRUCT_COUNTS = {128: (30, REFIT2_STRUCT_K), 120: (6, (2048, 4096, 5120, 8192)),
                        136: (6, (2048, 4096, 5120, 8192)), 64: (4, (1024, 4096, 8192)),
                        192: (4, (1024, 4096, 8192))}
REFIT2_STRUCT_M = (512, 32768)
REFIT2_STRUCT_N = (512, 131072)
# Best TFLOPS per quadrant tile in FULL (compute-bound), and TB/s, for the cost estimate (est_us) only.
# WORST10 (refit.py --study worst10): the AITER shapes where v27's auto tile is 10% or more behind the best library
# (experiments/aiter_lixun_shapes/results/worst10_shapes.csv, with control shapes where it is ahead), at every quadrant
# tile: every valid REFIT_KINDS point, the one-tile rs / ow points again with bf16 and bf16-pair partials (where the
# tile's partials check passed), and all. No atomic ending.
WORST10_CSV = os.path.join(A16, "..", "..", "..", "..", "experiments", "aiter_lixun_shapes", "results",
                           "worst10_shapes.csv")
WORST10_KINDS = (REFIT_KINDS + tuple(f"{e}{n}{p}" for e in ("rs", "ow") for n in REFIT_N + ("all",) for p in ("b", "p"))
                 + ("all",))
FULL_TFLOPS = {"256x256x64": 1529, "256x128x64": 1332, "128x256x64": 1274, "128x128x64": 1176, "128x128x128": 1142,
               "256x64x64": 938, "128x64x64": 887, "64x128x64": 875, "128x64x128": 852, "64x256x64": 826,
               "64x128x128": 802, "64x64x128": 729, "64x64x64": 689}
EST_TBPS = 4.5
FROZEN_FILES = ("v27_streamk_tile/matmul_kernel.py", "v27_streamk_tile/sweep.py", "v27_streamk_tile/refit.py",
                "v9_tile/matmul_kernel.py")
FIELDS = ("kind", "tile", "M", "N", "K", "rep", "correct", "error", "kernel", "policy", "fixup", "streamk_tiles",
          "tiles", "S", "waves", "k_steps", "n", "n_hi", "ending", "programs", "pairs_per_prog", "busy_cus",
          "us_median", "tflops", "tbps", "grid", "gpu", "round", "chunk", "pos", "ts", "iters")
FULL_SUFFIX = "-full"
SPLIT_FIXUP = {"rs": "rs", "ow": "owner", "az": "atomic_iz"}
PARTIALS_SUFFIX = {"b": "bf16", "p": "bf16_pairs"}


def pinned_spec(bm, bn, bk):
    """The small tile's spec with the fewest waves (from v9_tile's default up) whose W-wave workgroups
    fit 4 / W to a CU, as v27 requires (32x64x128: w2)."""
    sys.path.insert(0, os.path.join(A16, "..", "..", "utils"))
    from v9_tile.matmul_kernel import LDS_PER_CU, small_config, small_lds_bytes

    km, kn, bk, w, _ = small_config(bm, bn, bk)
    while small_lds_bytes(km, kn, bk) > LDS_PER_CU // (4 // w):
        w *= 2
    return f"{bm}x{bn}x{bk}" + ("" if w == small_config(bm, bn, bk)[3] else f"w{w}")


def v27_module():
    """matmul_kernel, reading its options (os.environ.get only) from a private copy of the environment, which runner
    sets per launch. Setting the process environment instead (setenv) races with rocprofv3's threads reading it:
    glibc's setenv is not thread-safe, and the timed workers segfaulted at random points."""
    import importlib
    import types

    import bench_ext  # noqa: F401  (paths)

    mod = importlib.import_module("v27_streamk_tile.matmul_kernel")
    if not isinstance(mod.os, types.SimpleNamespace):
        mod.os = types.SimpleNamespace(environ=dict(os.environ))
    return mod


def shape_info(tile, M, N, K):
    """The tile's view of a shape: tiles, S = tiles % programs, P = programs, pairs and k-steps per
    tile (K rounded up to pairs, as v27 runs it), on the kernel's dimensions."""
    v27 = v27_module()
    geo = v27.tile_geometry(tile)
    bm, bn, bk = geo["block_m"], geo["block_n"], geo["block_k"]
    m, n = (N, M) if geo["swap"] else (M, N)
    tiles = -(-m // bm) * -(-n // bn)
    P = geo["num_programs"]
    pairs = -(-K // (2 * bk))
    return dict(tiles=tiles, S=tiles % P, P=P, pairs=pairs, k_steps=2 * pairs, waves=tiles // P, small=geo["small"],
                block_n=bn)


def base_kind(kind):
    """(kind without -full, whether it runs on the full grid: STREAMK_COMPACT_GRID=0)."""
    if kind.endswith(FULL_SUFFIX):
        return kind[:-len(FULL_SUFFIX)], True
    return kind, False


def split_kind(kind):
    """('rs' | 'ow' | 'az', n or None for all, partials) for a one-tile split kind, else None."""
    kind = base_kind(kind)[0]
    if kind[:2] in SPLIT_FIXUP and kind != "all":
        partials = PARTIALS_SUFFIX.get(kind[-1], "fp32")
        n = kind[2:].rstrip("".join(PARTIALS_SUFFIX))
        return kind[:2], (None if n == "all" else int(n)), partials
    return None


def pdp_programs(kind):
    """P of a pdp{P} kind, else None."""
    m = re.fullmatch(r"pdp(\d+)", base_kind(kind)[0])
    return int(m.group(1)) if m else None


def full_spec(tile):
    """The spec with its BK (128x128 -> 128x128x64)."""
    bm, bn, bk = v27_module().parse_spec(tile)[:3]
    return f"{bm}x{bn}x{bk}"


def valid(kind, tile, M, N, K):
    kind = base_kind(kind)[0]
    if pdp_programs(kind):
        return pdp_programs(kind) <= PDP_MAX_PROGRAMS.get(full_spec(tile), 256)
    sh = shape_info(tile, M, N, K)
    s, P, pairs = sh["S"], sh["P"], sh["pairs"]
    if kind in ("dp", "pdp", "cap"):
        return True
    if s == 0:
        return False
    if kind == "2t":
        return sh["tiles"] >= P
    if kind == "sp":
        return s * pairs >= P
    if kind == "all":
        return True
    ending, n, partials = split_kind(kind)
    if partials != "fp32" and (ending == "az" or (partials == "bf16_pairs" and (sh["small"] or sh["block_n"] < 128))):
        return False
    n_lo, n_hi = min(P // s, pairs), min(P // s + 1, pairs)
    extra = P % s if n_hi > n_lo else 0
    if n_lo < 2:
        return False
    if n is None:
        return True
    return 2 <= n <= n_lo and not (n == n_lo and extra == 0)


def point_name(kind, tile, M, N, K, rep=False):
    return f"{kind}:{tile}:{M}x{N}x{K}" + (":r" if rep else "")


def prelim_points():
    """(points, repeats): every valid PRELIM_KINDS point, in a seeded shuffled order (so shards mix tiles
    and shapes), and the seeded NOISE_SHARE of them to be timed again."""
    grids = ((WQ_TILES, PRELIM_CB), (WQ_TILES + SMALL_TILES, PRELIM_MB))
    pts = [point_name(kind, t, M, N, K) for tiles, shapes in grids for (M, N, K) in shapes for t in tiles
           for kind in PRELIM_KINDS if valid(kind, t, M, N, K)]
    rng = random.Random(27)
    rng.shuffle(pts)
    reps = sorted(rng.sample(range(len(pts)), round(NOISE_SHARE * len(pts))))
    return pts, reps


def candidate_tiles():
    """Every tile spec v27 accepts: quadrant BM, BN in 64-256 and BK 64 / 128; a small side of 1-32 (as BM, or as BN,
    which runs C^T) with the other side 16-256 and BK 64-256."""
    v27 = v27_module()
    specs = [f"{m}x{n}x{k}" for m in (64, 128, 256) for n in (64, 128, 256) for k in (64, 128)]
    specs += [f"{m}x{n}x{k}" for m in (1, 2, 4, 8, 16, 32) for n in (16, 32, 64, 128, 256) for k in (64, 128, 256)]
    specs += [f"{m}x{n}x{k}" for n in (1, 2, 4, 8, 16, 32) for m in (64, 128, 256) for k in (64, 128, 256)]
    out = []
    for s in specs:
        try:
            v27.tile_geometry(s)
        except AssertionError:
            continue
        out.append(s)
    return out


def check_passed(path):
    """Whether a check_streamk.py report ends in PASS; a full check whose only failures are atomic_iz rows counts if
    its atomic re-run (<path>_atomic.md, §5: after the ATOMIC_EVEN fix) passed."""
    lines = open(path).read().strip().splitlines()
    if lines and lines[-1].strip() == "PASS":
        return True
    bad = [ln for ln in lines if ln.startswith("| ") and len(ln.split("|")) > 10
           and ("error:" in ln or ln.split("|")[8].strip() not in ("0", "wrong results", "---")
                or ln.split("|")[9].strip() not in ("0", "flags or counts left up", "---"))]
    rerun = path[:-3] + "_atomic.md"
    return (bool(bad) and all("atomic_iz" in ln for ln in bad) and os.path.exists(rerun)
            and check_passed(rerun))


def stress_checked(tile):
    """Whether the tile's stress check passed: the full one (results/check_streamk_<tile>.md, also under the spec
    without its default x64), or one restricted to FULL's configurations (results/check_sweep_<tile>.md)."""
    names = [tile] + ([tile[:-3]] if tile.endswith("x64") else [])
    for prefix in ("check_streamk", "check_sweep"):
        for n in names:
            path = os.path.join(HERE, "results", f"{prefix}_{n}.md")
            if os.path.exists(path):
                return check_passed(path)
    return False


def full_points():
    """(points, repeats) of FULL: quadrant tiles on FULL_CB, every stress-checked tile on FULL_MB."""
    tiles = [t for t in candidate_tiles() if stress_checked(t)]
    quad = [t for t in tiles if not shape_info(t, 1024, 1024, 4096)["small"]]
    grids = ((quad, FULL_CB), (tiles, FULL_MB))
    pts = [point_name(kind, t, M, N, K) for ts, shapes in grids for (M, N, K) in shapes for t in ts
           for kind in FULL_KINDS if valid(kind, t, M, N, K)]
    rng = random.Random(27)
    rng.shuffle(pts)
    reps = sorted(rng.sample(range(len(pts)), round(FULL_NOISE_SHARE * len(pts))))
    return pts, reps


def aiter_below_one_wave(count, seed=27):
    """A seeded sample of AITER's distinct bf16 shapes with fewer 256x256 tiles than 256 (at least one)."""
    seen = []
    for r in csv.DictReader(open(AITER_SHAPES_CSV)):
        M, N, K = int(r["M"]), int(r["N"]), int(r["K"])
        if (M, N, K) not in seen and 0 < -(-M // 256) * -(-N // 256) < 256:
            seen.append((M, N, K))
    return sorted(random.Random(seed).sample(seen, min(count, len(seen))))


def compact_sets():
    """{set name: [(kind, tile, M, N, K)]} of the COMPACT preset."""
    mb = [(k, pinned_spec(m, bn, 128), m, n, 8192) for m in MB_M for n in MB_N for bn in MB_BN for k in COMPACT_KINDS]
    wq = [(k, t, 4096, n, kk) for t in WQ_TILES for n in COMPACT_WQ_N for kk in (4096, 8192) for k in COMPACT_KINDS
          if shape_info(t, 4096, n, kk)["waves"] == 0]
    aiter = [(k, t, M, N, K) for (M, N, K) in aiter_below_one_wave(COMPACT_AITER) for t in COMPACT_AITER_TILES
             for k in COMPACT_KINDS]
    return {"MB": mb, "WQ": wq, "AITER": aiter}


def compact_points(shard=None):
    """Every COMPACT point as full, compact, full again, compact again (consecutive, so each pair of modes is
    timed close together, and the repeats give the noise band), in a seeded order, whole groups per shard."""
    base = sorted({p for pts in compact_sets().values() for p in pts if valid(*p)})
    random.Random(27).shuffle(base)
    if shard is not None:
        base = base[shard[0]::shard[1]]
    out = []
    for kind, tile, M, N, K in base:
        f, c = point_name(kind + FULL_SUFFIX, tile, M, N, K), point_name(kind, tile, M, N, K)
        out += [f, c, f + ":r", c + ":r"]
    return out


def heldout_shapes():
    """The held-out test: AITER's distinct real-model shapes (AITER_SHAPES_CSV), never in the fitting grid."""
    return sorted({(int(r["M"]), int(r["N"]), int(r["K"])) for r in csv.DictReader(open(AITER_SHAPES_CSV))})


def log_uniform(rng, lo, hi):
    return 2 ** rng.uniform(math.log2(lo), math.log2(hi))


def refit_k(rng, k, lo, hi):
    """Half the shapes of a K bin at the bin's power of two, half log-uniform within half an octave of it (multiples
    of 256, inside [lo, hi])."""
    if rng.random() < 0.5:
        return k
    return min(hi, max(lo, round(log_uniform(rng, k / 2 ** 0.5, k * 2 ** 0.5) / 256) * 256))


@functools.lru_cache(None)
def refit_shapes():
    """(compute-bound, memory-bound) shapes of REFIT, seeded (REFIT_SEED), none of them held out.
    Compute-bound: per cell of REFIT_CB_WAVES x REFIT_CB_S (in 256x256 tiles over 256 programs), REFIT_CB_PER_CELL
    shapes, each K bin of REFIT_CB_K three times and one more at random; waves, leftover tiles (log-uniform) and
    M / 256 (log-uniform in 2-32) drawn, N / 256 = tiles / (M / 256) rounded, kept if the shape lands in its cell.
    Memory-bound: per M in REFIT_MB_M, REFIT_MB_PER_M shapes with log N in equal strata of REFIT_MB_N (one draw each)
    and the K bins of REFIT_MB_K dealt evenly in a shuffled order."""
    rng = random.Random(REFIT_SEED)
    held = set(heldout_shapes())
    seen = set()
    cb = []
    for w0, w1 in REFIT_CB_WAVES:
        for s0, s1 in REFIT_CB_S:
            bins = list(range(len(REFIT_CB_K))) * 3 + [rng.randrange(len(REFIT_CB_K))]
            rng.shuffle(bins)
            for kb in bins:
                while True:
                    tiles = 256 * rng.randint(w0, w1) + min(s1 - 1, max(s0, round(log_uniform(rng, s0, s1))))
                    mt = round(log_uniform(rng, 2, 32))
                    nt = round(tiles / mt)
                    t = mt * nt
                    K = refit_k(rng, REFIT_CB_K[kb], REFIT_CB_K[0], REFIT_CB_K[-1])
                    sh = (256 * mt, 256 * nt, K)
                    if (1 <= nt <= 64 and w0 <= t // 256 <= w1 and s0 <= t % 256 < s1 and sh not in held
                            and sh not in seen):
                        break
                seen.add(sh)
                cb.append(sh)
    mb = []
    lo, hi = (math.log2(x) for x in REFIT_MB_N)
    for M in REFIT_MB_M:
        kb = [i % len(REFIT_MB_K) for i in range(REFIT_MB_PER_M)]
        rng.shuffle(kb)
        for i in range(REFIT_MB_PER_M):
            while True:
                x = lo + (hi - lo) * (i + rng.random()) / REFIT_MB_PER_M
                N = max(256, round(2 ** x / 256) * 256)
                K = refit_k(rng, REFIT_MB_K[kb[i]], REFIT_MB_K[0], REFIT_MB_K[-1])
                if (M, N, K) not in seen and (M, N, K) not in held:
                    break
            seen.add((M, N, K))
            mb.append((M, N, K))
    return cb, mb


def stamp_hash():
    """sha256 of matmul_kernel.py, the stamp on a REFIT check report."""
    return hashlib.sha256(open(os.path.join(HERE, "matmul_kernel.py"), "rb").read()).hexdigest()


def refit_check_path(tile, programs=None):
    return os.path.join(HERE, "results", "refit_checks",
                        f"check_sweep_{tile}" + (f"_pdp{programs}" if programs else "") + ".md")


def refit_checked(tile, programs=None):
    """Whether the tile's REFIT stress check (tmp/check_queue.py with STAMP, results/refit_checks/; programs: the
    pdp{P} check) passed on the current matmul_kernel.py: its first line is the hash it ran on (stamp_hash)."""
    path = refit_check_path(tile, programs)
    if not os.path.exists(path):
        return False
    first = open(path).readline().split()
    return first[-1:] == [stamp_hash()] and check_passed(path)


def refit_points(checked=True):
    """{(M, N, K): [points]} of REFIT, every valid REFIT_KINDS point at the REFIT_TILES (checked: those that passed
    refit_checked, and pdp{P} only where its check passed too): quadrant tiles on the compute-bound shapes, all tiles
    on the memory-bound ones."""
    tiles = [t for t in REFIT_TILES if not checked or refit_checked(t)]
    quad = [t for t in tiles if not shape_info(t, 1024, 1024, 4096)["small"]]
    cb, mb = refit_shapes()
    out = {}
    for ts, shapes in ((quad, cb), (tiles, mb)):
        for sh in shapes:
            out[sh] = [point_name(k, t, *sh) for t in ts for k in REFIT_KINDS if valid(k, t, *sh)
                       and not (checked and pdp_programs(k) and not refit_checked(t, pdp_programs(k)))]
    return out


def capped_k(rng, kb, bins, cap):
    """refit_k at bin kb of bins, the bin lowered to the largest one at most cap (the largest K a cost limit allows,
    such as limit / (M N)) and K kept within it; None if even the smallest bin is over cap."""
    ok = [i for i, k in enumerate(bins) if k <= cap]
    if not ok:
        return None
    kb = min(kb, ok[-1])
    return min(refit_k(rng, bins[kb], bins[0], bins[-1]), int(cap) // 256 * 256)


def cell_reachable(m_rng, n_rng, w, s):
    """Whether some M / 256 in m_rng and N / 256 in n_rng give a tile count in the waves x leftover cell."""
    lo, hi = 256 * w[0] + s[0], 256 * w[1] + s[1] - 1
    return m_rng[0] * n_rng[0] <= hi and m_rng[1] * n_rng[1] >= lo


@functools.lru_cache(None)
def refit2_shapes():
    """{group: [(M, N, K)]} of REFIT2: refit2_base_shapes() and the structured group (refit2_structured)."""
    return {**refit2_base_shapes(), "structured": sorted(refit2_structured())}


@functools.lru_cache(None)
def refit2_base_shapes():
    """{group: [(M, N, K)]} of REFIT2 before the structured group, seeded (REFIT2_SEED), none of REFIT's or the
    held-out set's.
    Compute-bound (REFIT2_CB_GROUPS): per reachable cell, each K bin of REFIT_CB_K dealt evenly (shuffled); waves and
    leftover tiles drawn as REFIT (leftover uniform below 4, else log-uniform), M / 256 log-uniform in the group's
    range, N / 256 = tiles / (M / 256) rounded, kept if in range and in the cell; K lowered to fit REFIT2_CB_MNK.
    Memory-bound: per M in REFIT2_MB_M, REFIT2_MB_PER_M shapes with log N in equal strata of REFIT2_MB_N and the bins
    of REFIT2_MB_K dealt evenly, K lowered to fit REFIT2_MB_NK."""
    rng = random.Random(REFIT2_SEED)
    cb_old, mb_old = refit_shapes()
    seen = set(heldout_shapes()) | set(cb_old) | set(mb_old)
    out = {}
    for name, m_rng, n_rng, waves, lefts, per_cell in REFIT2_CB_GROUPS:
        out[name] = []
        for w0, w1 in waves:
            for s0, s1 in lefts:
                if not cell_reachable(m_rng, n_rng, (w0, w1), (s0, s1)):
                    continue
                bins = [i % len(REFIT_CB_K) for i in range(per_cell)]
                rng.shuffle(bins)
                for kb in bins:
                    for _ in range(100000):
                        s = rng.randint(s0, s1 - 1) if s0 == 0 else min(s1 - 1, max(s0, round(log_uniform(rng, s0, s1))))
                        tiles = 256 * rng.randint(w0, w1) + s
                        mt = round(log_uniform(rng, *m_rng))
                        nt = round(tiles / mt)
                        t = mt * nt
                        if not (n_rng[0] <= nt <= n_rng[1] and w0 <= t // 256 <= w1 and s0 <= t % 256 < s1):
                            continue
                        K = capped_k(rng, kb, REFIT_CB_K, REFIT2_CB_MNK / (256 * mt * 256 * nt))
                        sh = (256 * mt, 256 * nt, K)
                        if K and sh not in seen:
                            break
                    else:
                        raise RuntimeError(f"REFIT2 {name}: no shape found in cell waves {w0}-{w1}, S {s0}-{s1}")
                    seen.add(sh)
                    out[name].append(sh)
    out["memory-bound"] = []
    lo, hi = (math.log2(x) for x in REFIT2_MB_N)
    for M in REFIT2_MB_M:
        kb = [i % len(REFIT2_MB_K) for i in range(REFIT2_MB_PER_M)]
        rng.shuffle(kb)
        for i in range(REFIT2_MB_PER_M):
            while True:
                x = lo + (hi - lo) * (i + rng.random()) / REFIT2_MB_PER_M
                N = max(256, round(2 ** x / 256) * 256)
                K = capped_k(rng, kb[i], REFIT2_MB_K, REFIT2_MB_NK / N)
                if (M, N, K) not in seen:
                    break
            seen.add((M, N, K))
            out["memory-bound"].append((M, N, K))
    return out


@functools.lru_cache(None)
def refit2_structured():
    """{(M, N, K): (BM, BN)} of REFIT2's structured group (REFIT2_STRUCT_*), none of REFIT's, the held-out set's or
    refit2_base_shapes()'. Per target and tile count, the factorisations (M / BM) (N / BN) = count within
    REFIT2_STRUCT_M / _N are cycled in a seeded order against the count's K values (shuffled), so every K and
    orientation appears."""
    rng = random.Random(REFIT2_SEED + 2)
    cb_old, mb_old = refit_shapes()
    seen = set(heldout_shapes()) | set(cb_old) | set(mb_old) | {sh for v in refit2_base_shapes().values() for sh in v}
    out = {}
    for bm, bn in REFIT2_STRUCT_TARGETS:
        for count, (n, ks) in REFIT2_STRUCT_COUNTS.items():
            geos = [(mt * bm, count // mt * bn) for mt in range(1, count + 1) if count % mt == 0
                    and REFIT2_STRUCT_M[0] <= mt * bm <= REFIT2_STRUCT_M[1]
                    and REFIT2_STRUCT_N[0] <= count // mt * bn <= REFIT2_STRUCT_N[1]]
            rng.shuffle(geos)
            kk = [ks[i % len(ks)] for i in range(n)]
            rng.shuffle(kk)
            i = 0
            for K in kk:
                for _ in range(len(geos)):
                    M, N = geos[i % len(geos)]
                    i += 1
                    if (M, N, K) not in seen:
                        break
                else:
                    continue
                seen.add((M, N, K))
                out[(M, N, K)] = (bm, bn)
    return out


def structured_tiles(bm, bn):
    """The REFIT quadrant tiles a structured target is timed at: its own (both BK), those of half and a quarter its
    area, 256x256x64 and 128x128x64."""
    quad = [t for t in REFIT_TILES if not shape_info(t, 1024, 1024, 4096)["small"]]
    area = lambda t: v27_module().parse_spec(t)[0] * v27_module().parse_spec(t)[1]  # noqa: E731
    return [t for t in quad if area(t) in (bm * bn, bm * bn // 2, bm * bn // 4)
            and (area(t) != bm * bn or v27_module().parse_spec(t)[:2] == (bm, bn))
            or t in ("256x256x64", "128x128x64")]


@functools.lru_cache(None)
def refit2_test():
    """REFIT2's test split: a seeded REFIT2_TEST_SHARE of each compute-bound group and of each memory-bound M, and
    (from its own seed, so the other groups' split is unchanged) of each structured target."""
    rng = random.Random(REFIT2_SEED + 1)
    strata = {k: v for k, v in refit2_shapes().items() if k not in ("memory-bound", "structured")}
    for sh in refit2_shapes()["memory-bound"]:
        strata.setdefault(f"memory-bound M={sh[0]}", []).append(sh)
    test = {sh for _, v in sorted(strata.items()) for sh in rng.sample(v, round(REFIT2_TEST_SHARE * len(v)))}
    rng = random.Random(REFIT2_SEED + 3)
    by_target = collections.defaultdict(list)
    for sh, target in sorted(refit2_structured().items()):
        by_target[target].append(sh)
    test |= {sh for _, v in sorted(by_target.items()) for sh in rng.sample(v, round(REFIT2_TEST_SHARE * len(v)))}
    return frozenset(test)


def raggedm_checked(tile):
    """Whether a small tile's ragged-M check (check_queue.py line "TILE raggedm") passed on the current kernel."""
    path = os.path.join(HERE, "results", "refit_checks", f"check_sweep_{tile}_raggedm.md")
    if not os.path.exists(path):
        return False
    first = open(path).readline().split()
    return first[-1:] == [stamp_hash()] and check_passed(path)


def refit2_points(checked=True):
    """{(M, N, K): [points]} of REFIT2: as refit_points, quadrant tiles on the compute-bound groups (the structured
    group: its target's structured_tiles), every tile on the memory-bound shapes less REFIT2_MB_PRUNE (checked: small
    tiles also need raggedm_checked)."""
    tiles = [t for t in REFIT_TILES if not checked or refit_checked(t)]
    quad = [t for t in tiles if not shape_info(t, 1024, 1024, 4096)["small"]]
    small = [t for t in tiles if t not in quad and (not checked or raggedm_checked(t))]
    out = {}
    for g, shapes in refit2_shapes().items():
        for sh in shapes:
            if g == "structured":
                ts = [t for t in structured_tiles(*refit2_structured()[sh]) if t in quad]
            elif g == "memory-bound":
                ts = [t for t in quad + small if sh[0] < REFIT2_MB_PRUNE.get(t, 1 << 30)]
            else:
                ts = quad
            out[sh] = [point_name(k, t, *sh) for t in ts for k in REFIT_KINDS if valid(k, t, *sh)
                       and not (checked and pdp_programs(k) and not refit_checked(t, pdp_programs(k)))]
    return out


def partials_checked(tile):
    """Whether the tile's bf16 / bf16-pair partials check (check_queue.py line "TILE partials") passed on the current
    kernel."""
    path = os.path.join(HERE, "results", "refit_checks", f"check_sweep_{tile}_partials.md")
    if not os.path.exists(path):
        return False
    first = open(path).readline().split()
    return first[-1:] == [stamp_hash()] and check_passed(path)


def worst10_shapes():
    """{(M, N, K): group} of WORST10_CSV ("worst" or "control"); a shape in both output dtypes appears once (the
    sweep times bf16 output)."""
    out = {}
    for r in csv.DictReader(open(WORST10_CSV)):
        out.setdefault((int(r["M"]), int(r["N"]), int(r["K"])), r["group"])
    return out


def worst10_points(checked=True):
    """{(M, N, K): [points]} of WORST10: every valid WORST10_KINDS point at the quadrant REFIT_TILES (checked: those
    that passed refit_checked; pdp{P} only where its check passed; bf16 / bf16-pair partials only where
    partials_checked)."""
    quad = [t for t in REFIT_TILES if not shape_info(t, 1024, 1024, 4096)["small"] and (not checked or refit_checked(t))]
    out = {}
    for sh in worst10_shapes():
        pts = []
        for t in quad:
            for k in WORST10_KINDS:
                if not valid(k, t, *sh):
                    continue
                if checked and pdp_programs(k) and not refit_checked(t, pdp_programs(k)):
                    continue
                split = split_kind(k)
                if checked and split and split[2] != "fp32" and not partials_checked(t):
                    continue
                pts.append(point_name(k, t, *sh))
        out[sh] = pts
    return out


def est_us(tile, M, N, K):
    """A rough kernel time (us) for balancing shards and the run's estimate: FULL's best TFLOPS at a quadrant tile
    (400 at a small one), or EST_TBPS on unique bytes, whichever is slower, plus 3 us."""
    tf = FULL_TFLOPS.get(full_spec(tile), 400)
    return max(2 * M * N * K / (tf * 1e12), 2 * (M * K + K * N + M * N) / (EST_TBPS * 1e12)) * 1e6 + 3


def shape_iters(M, N, K):
    """Launches timed per point (--iters 0): 1000, or for a shape whose fastest kernel takes more than 250 us
    (est_us at 256x256x64) about 0.25 s of launches, at least 200. The same for every point of a shape."""
    t = est_us("256x256x64", M, N, K)
    return 1000 if t <= 250 else max(2 * GRAPH_LEN, GRAPH_LEN * math.ceil(2.5e5 / t / GRAPH_LEN))


def point_cost(p, iters=0):
    """Estimated seconds to time point p: FULL's 0.17 s per point outside the kernels, plus its launches."""
    kind, tile, M, N, K = parse_point(p)
    copies = max(1, ROTATING_MB * 2**20 // (2 * (M * K + K * N + M * N)))
    return 0.17 + (1 + copies + (iters or shape_iters(M, N, K))) * est_us(tile, M, N, K) * 1e-6


def file_hashes():
    """sha256 of each FROZEN_FILES file (relative to a16w16/)."""
    return {f: hashlib.sha256(open(os.path.join(A16, f), "rb").read()).hexdigest() for f in FROZEN_FILES}


def check_frozen(manifest):
    """Stop (exit 3) if a file the run manifest hashed has changed since."""
    want = json.load(open(manifest))["files"]
    now = file_hashes()
    changed = [f for f, h in want.items() if now.get(f) != h]
    if changed:
        print(f"FROZEN: {', '.join(changed)} changed since {manifest} was written; stopping", flush=True)
        sys.exit(3)


@functools.lru_cache(None)
def smi_index(gpu):
    """rocm-smi's index of HIP device gpu, matched by PCI address: the two number the GPUs differently (here HIP 0 is
    rocm-smi's 3, HIP 4 its 7). The address comes from a subprocess, so this process never opens the GPU."""
    code = "import torch; p = torch.cuda.get_device_properties(0); print(p.pci_domain_id, p.pci_bus_id, p.pci_device_id)"
    out = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu)),
                         capture_output=True, text=True, check=True, timeout=300).stdout.split()
    want = tuple(int(x) for x in out[-3:])
    smi = subprocess.run(["rocm-smi", "--showbus"], capture_output=True, text=True, timeout=120).stdout
    for line in smi.splitlines():
        m = re.match(r"GPU\[(\d+)\].*PCI Bus:\s*([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9a-fA-F]+)\.", line.strip())
        if m and tuple(int(x, 16) for x in m.groups()[1:]) == want:
            return m.group(1)
    raise RuntimeError(f"HIP device {gpu} (PCI {want}) is not in rocm-smi --showbus")


def gpu_pids(gpu):
    """PIDs (in the host's namespace) that rocm-smi --showpidgpus lists on HIP device gpu (smi_index)."""
    idx = smi_index(gpu)
    out = subprocess.run(["rocm-smi", "--showpidgpus"], capture_output=True, text=True, timeout=120).stdout
    pids, cur = [], None
    for line in out.splitlines():
        m = re.match(r"PID (\d+) is using", line.strip())
        if m:
            cur = m.group(1)
        elif cur and re.fullmatch(r"[\d\s]+", line.strip() or "x"):
            if idx in line.split():
                pids.append(cur)
            cur = None
    return pids


def foreign_pids(gpu, settle=3, wait=2.0):
    """Processes on the GPU while none of ours runs there, seen in each of `settle` looks `wait` s apart (a process
    of ours that just exited can stay listed for a moment)."""
    pids = None
    for i in range(settle):
        now = set(gpu_pids(gpu))
        pids = now if pids is None else pids & now
        if not pids:
            return []
        if i + 1 < settle:
            time.sleep(wait)
    return sorted(pids)


@contextlib.contextmanager
def gpu_lock(gpu):
    """check_streamk.py's and check_grid.py's per-GPU lock, so no stress check launches while we time."""
    with open(f"/tmp/v27_check_gpu{gpu}.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def wait_for_gpu(gpu, max_wait, log):
    """Wait until no foreign process is on the GPU, logging it; stop (exit 4) after max_wait seconds."""
    t0 = time.monotonic()
    while True:
        pids = foreign_pids(gpu)
        if not pids:
            return
        if time.monotonic() - t0 > max_wait:
            log(f"FOREIGN: pids {pids} still on GPU {gpu} (rocm-smi {smi_index(gpu)}) after {max_wait} s; stopping")
            sys.exit(4)
        log(f"FOREIGN: pids {pids} on GPU {gpu} (rocm-smi {smi_index(gpu)}); waiting")
        time.sleep(30)


def points_for(name, shard=None):
    """The points of a preset or a comma-separated list, those of shard (i, n) only if given: shard i
    takes every n-th point from i, then the repeats of shard i - 1's points."""
    if name == "COMPACT":
        return compact_points(shard)
    if name in ("REFIT", "REFIT2") or name.startswith("@"):
        # @FILE: every point of a run manifest's blocks (refit.py init).
        blocks = ({"REFIT": refit_points, "REFIT2": refit2_points}[name]() if not name.startswith("@")
                  else json.load(open(name[1:]))["blocks"])
        pts = [p for ps in blocks.values() for p in ps]
        return pts if shard is None else pts[shard[0]::shard[1]]
    if name in ("PRELIM", "FULL"):
        pts, reps = prelim_points() if name == "PRELIM" else full_points()
        if shard is None:
            return pts + [pts[j] + ":r" for j in reps]
        i, n = shard
        return pts[i::n] + [pts[j] + ":r" for j in reps if j % n == (i - 1) % n]
    if name == "MB":
        pts = [f"{kind}:{pinned_spec(m, bn, 128)}:{m}x{n}x8192" for m in MB_M for n in MB_N for bn in MB_BN
               for kind in ("dp", "cap", "all")]
    elif name in ("WQ4096", "WQ8192"):
        k = int(name[2:])
        pts = ([f"{kind}:{t}:4096x{n}x{k}" for t in WQ_TILES for n in WQ_N for kind in ("dp", "cap")]
               + [f"all:{t}:4096x{n}x{k}" for t in WQ_TILES for n in WQ_SK_N])
    else:
        pts = name.split(",")
    return pts if shard is None else pts[shard[0]::shard[1]]


KERNELS = ("v27_streamk_tile", "v9_tile", "v9_tile_small", "v9_tile_2r")


def parse_point(p):
    kind, tile, shape, *rep = p.split(":")
    M, N, K = (int(x) for x in shape.split("x"))
    return kind, tile, M, N, K


def is_rep(p):
    return p.endswith(":r")


def kind_env(kind, tile, M, N, K):
    """The STREAMK_* settings of a v27 kind (-full: on the full grid, STREAMK_COMPACT_GRID=0)."""
    kind, full = base_kind(kind)
    if full:
        return {**kind_env(kind, tile, M, N, K), "STREAMK_COMPACT_GRID": "0"}
    sh = shape_info(tile, M, N, K)
    if kind == "cap":
        return {}
    if kind == "pdp":
        return {"STREAMK_POLICY": "dp"}
    if pdp_programs(kind):
        return {"STREAMK_POLICY": "dp", "STREAMK_PERSISTENT_PROGRAMS": str(pdp_programs(kind))}
    if kind == "all":
        return {"STREAMK_POLICY": "one_tile", "STREAMK_NUM_PROGRAMS": str(sh["P"])}
    if kind == "2t":
        return {"STREAMK_POLICY": "two_tile"}
    if kind == "sp":
        return {"STREAMK_POLICY": "spread"}
    ending, n, partials = split_kind(kind)
    return {"STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": SPLIT_FIXUP[ending], "STREAMK_PARTIALS": partials,
            "STREAMK_NUM_PROGRAMS": str(sh["P"] if n is None else sh["S"] * n)}


def runner(kind, tile, M, N, K):
    v27 = v27_module()
    opts = v27.os.environ
    for k in ("STREAMK_POLICY", "STREAMK_NUM_PROGRAMS", "STREAMK_FIXUP", "STREAMK_PARTIALS", "STREAMK_COMPACT_GRID",
              "STREAMK_PERSISTENT_PROGRAMS"):
        opts.pop(k, None)
    if kind == "dp":
        import importlib

        mk = importlib.import_module("v9_tile.matmul_kernel")
        spec = v27.fallback_spec(tile, v27.tile_geometry(tile))
        return lambda a, b, c: mk.matmul(a, b, c, tile=spec)
    env = kind_env(kind, tile, M, N, K)

    def run(a, b, c):
        opts.update(env)
        v27.matmul(a, b, c, tile=tile)
        for k in env:
            opts.pop(k, None)
    return run


def derive(launch, sh):
    """The split columns of a row from what ran (LAST_LAUNCH) and the tile's view of the shape."""
    P, pairs = launch.get("num_programs") or sh["P"], sh["pairs"]
    sk = launch.get("streamk_tiles") or 0
    if launch["kernel"] != "v27_streamk_tile" or launch["policy"] == "dp" or sk == 0:
        progs = sh["tiles"] % P or P
        return dict(n=1, n_hi=1, ending="dp", programs=progs, pairs_per_prog=pairs, busy_cus=min(progs, 256))
    if launch["policy"] == "spread" or sk > P:
        progs = min(P, sk * pairs)
        return dict(n=round(P / sk, 2), n_hi=round(P / sk, 2), ending="e2e", programs=progs,
                    pairs_per_prog=round(sk * pairs / P, 2), busy_cus=min(progs, 256))
    spr = launch["streamk_programs"]
    n_lo, n_hi = min(spr // sk, pairs), min(spr // sk + 1, pairs)
    extra = spr % sk if n_hi > n_lo else 0
    progs = extra * n_hi + (sk - extra) * n_lo
    return dict(n=n_lo, n_hi=n_hi if extra else n_lo, ending=launch["fixup"], programs=progs,
                pairs_per_prog=round(sk * pairs / progs, 2), busy_cus=min(progs, 256))


def worker(args):
    import bench_ext as bench
    import torch

    v27 = v27_module()
    manifest = []
    for p in args.points.split(","):
        kind, tile, M, N, K = parse_point(p)
        iters = args.iters or shape_iters(M, N, K)
        run = runner(kind, tile, M, N, K)
        dt = torch.bfloat16
        a = torch.rand((M, K), device="cuda", dtype=dt) - 0.5
        b = bench.make_b(N, K, dt, False, lambda t: t.uniform_(-0.5, 0.5))
        c = bench.make_c(M, N, dt, False)
        run(a, b, c)
        launched = dict(v27.LAST_LAUNCH) if kind != "dp" else {"kernel": "v9_tile"}
        ref = a.float() @ b.float()
        split = split_kind(kind)
        rounds = launched.get("fixup") == "atomic_iz" or (split is not None and split[2] != "fp32")
        if rounds and launched.get("streamk_tiles"):
            # Every partial is rounded (into C, or to bf16): check_streamk.py's mean-error test.
            from v27_streamk_tile.check_streamk import tile_mean

            bm, bn = v27.parse_spec(tile)[:2]
            reps = -(-1024 // (min(M, bm) * min(N, bn)))
            bm, bn = (bm, bn * reps) if bn >= bm else (bm * reps, bn)
            sk = launched["streamk_tiles"]
            n_max = min(launched["streamk_programs"] // sk + 1, -(-K // (2 * v27.tile_geometry(tile)["block_k"])))
            rnd = tile_mean((ref.to(dt).float() - ref).abs(), bm, bn)
            err = tile_mean((c.float() - ref).abs(), bm, bn)
            correct = not bool(torch.isnan(c).any() or (err > (1 + n_max ** 0.5) * rnd + 1e-3).any())
        else:
            correct = bool(torch.allclose(c, ref.to(dt), atol=1e-1, rtol=1e-2))
        del a, b, c, ref
        a_l, b_l, c_l, copies = bench.gen_rotating_tensors(M, N, K, dt, ROTATING_MB)
        for i in range(copies):
            run(a_l[i], b_l[i], c_l[i])
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(GRAPH_LEN):
                run(a_l[i % copies], b_l[i % copies], c_l[i % copies])
        for _ in range(iters // GRAPH_LEN):
            g.replay()
        torch.cuda.synchronize()
        del g, a_l, b_l, c_l
        torch.cuda.empty_cache()
        manifest.append({"point": p, "dispatches": 1 + copies + (iters // GRAPH_LEN) * GRAPH_LEN,
                         "correct": correct, "iters": iters, "launch": {k: launched.get(k) for k in (
                             "kernel", "policy", "fixup", "streamk_tiles", "streamk_programs", "grid",
                             "num_programs")}})
        print(f"{p} copies={copies} correct={correct}", flush=True)
    json.dump(manifest, open(args.manifest, "w"), indent=1)


def warm_worker(args):
    """Compile the v27 build of every point (matmul_kernel.WARMUP: no launch) into Triton's cache; dp
    points and capped's v9_tile fallbacks are left to the timing run (small builds)."""
    import bench_ext as bench
    import torch

    v27 = v27_module()
    v27.WARMUP = True
    for p in args.points.split(","):
        kind, tile, M, N, K = parse_point(p)
        try:
            run = runner(kind, tile, M, N, K)
            a = torch.empty((M, K), device="cuda", dtype=torch.bfloat16)
            b = bench.make_b(N, K, torch.bfloat16, False, lambda t: None)
            run(a, b, bench.make_c(M, N, torch.bfloat16, False))
            print(f"WARM_OK {p}", flush=True)
        except Exception as e:  # noqa: BLE001  (reported, and the point fails again when timed)
            print(f"WARM_FAIL {p} {type(e).__name__}: {e}", flush=True)


def warm(points, jobs, gpus, fail_out=None):
    """Compile the points' v27 builds in `jobs` processes over `gpus`; fail_out: write {point: error} of those that
    failed (refit.py leaves them out of the timing)."""
    from concurrent.futures import ThreadPoolExecutor

    todo = sorted({p[:-2] if is_rep(p) else p for p in points if parse_point(p)[0] != "dp"})
    per = max(1, -(-len(todo) // (4 * jobs)))
    chunks = [todo[i:i + per] for i in range(0, len(todo), per)]
    fails, done = [], [0]
    lock = threading.Lock()

    def one(i):
        env = dict(os.environ, HIP_VISIBLE_DEVICES=gpus[i % len(gpus)])
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "warm_worker", "--points", ",".join(chunks[i])],
                           cwd=A16, env=env, capture_output=True, text=True)
        ok = sum(line.startswith("WARM_OK") for line in p.stdout.splitlines())
        with lock:
            fails.extend(line for line in p.stdout.splitlines() if line.startswith("WARM_FAIL"))
            if p.returncode:
                fails.append(f"WARM_FAIL chunk {i} exited {p.returncode}: {p.stderr[-500:]}")
            done[0] += ok
            print(f"warm: {done[0]} / {len(todo)}", flush=True)

    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(one, range(len(chunks))))
    for f in fails:
        print(f, flush=True)
    print(f"warm: {done[0]} of {len(todo)} compiled, {len(fails)} failures", flush=True)
    if fail_out:
        bad = dict(f[len("WARM_FAIL "):].split(" ", 1) for f in fails if not f.startswith("WARM_FAIL chunk"))
        json.dump({"failed": bad, "chunks": [f for f in fails if f.startswith("WARM_FAIL chunk")]},
                  open(fail_out, "w"), indent=1)


def timing(points, out_dir, iters=1000, point_timeout=300):
    """Time points in one rocprofv3 process. Returns (rows, None) or (None, index of the point that
    failed or hung: no output for point_timeout seconds)."""
    import glob
    import shutil

    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    manifest = os.path.join(out_dir, "manifest.json")
    cmd = ["rocprofv3", "--kernel-trace", "--kernel-include-regex", "|".join(KERNELS), "-d", out_dir, "-o", "run",
           "--output-format", "csv", "--", sys.executable, os.path.abspath(__file__), "worker", "--iters", str(iters),
           "--manifest", manifest, "--points", ",".join(points)]
    # env: os.environ, not the process's C environment, which the ROCm libraries loaded with torch have
    # changed (ROCPROFILER_REGISTER_LIBRARY: rocprofv3 then aborts).
    proc = subprocess.Popen(cmd, cwd=A16, env=dict(os.environ), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(x) for x in proc.stdout] + [lines.put(None)], daemon=True).start()
    done, tail, last, finished = 0, [], time.monotonic(), []
    while True:
        try:
            line = lines.get(timeout=max(0.0, last + point_timeout - time.monotonic()))
        except queue.Empty:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            print(f"timeout after {point_timeout} s on {points[min(done, len(points) - 1)]}", file=sys.stderr)
            return None, min(done, len(points) - 1)
        if line is None:
            break
        tail = (tail + [line])[-40:]
        if " copies=" in line:
            done, last = done + 1, time.monotonic()
            finished.append(time.time())
    if proc.wait():
        print("".join(tail), file=sys.stderr)
        return None, min(done, len(points) - 1)
    man = json.load(open(manifest))
    files = glob.glob(os.path.join(out_dir, "**", "*kernel_trace.csv"), recursive=True)
    rows = [r for r in csv.DictReader(open(files[0])) if r["Kernel_Name"] in KERNELS]
    rows.sort(key=lambda r: int(r["Dispatch_Id"]))
    try:
        per_point = base.chunks(rows, man)
    except AssertionError as e:
        print(e, file=sys.stderr)
        return None, len(points) - 1
    res = []
    for i, (m, chunk) in enumerate(zip(man, per_point)):
        kind, tile, M, N, K = parse_point(m["point"])
        us = [(int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3 for r in chunk[-100:]]
        med = S.median(us)
        r = row(m["point"], m["launch"], correct=m["correct"])
        r.update(us_median=round(med, 3), tflops=round(2 * M * N * K / (med * 1e-6) / 1e12, 1),
                 tbps=round(2 * (M * K + K * N + M * N) / (med * 1e-6) / 1e12, 3), iters=m.get("iters", ""),
                 ts=round(finished[i], 1) if i < len(finished) else "")
        res.append(r)
    return res, None


def row(p, launch, correct="", error=""):
    kind, tile, M, N, K = parse_point(p)
    sh = shape_info(tile, M, N, K)
    r = dict.fromkeys(FIELDS, "")
    r.update(kind=kind, tile=tile, M=M, N=N, K=K, rep=int(is_rep(p)), correct=correct, error=error,
             tiles=sh["tiles"], S=sh["S"], waves=sh["waves"], k_steps=sh["k_steps"])
    if launch:
        r.update({k: launch.get(k) for k in ("kernel", "policy", "fixup", "streamk_tiles", "grid")})
        r.update(derive(launch, sh))
    return r


def point_key(r):
    return point_name(r["kind"], r["tile"], r["M"], r["N"], r["K"], rep=str(r.get("rep", "0")) == "1")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="timing", choices=["timing", "worker", "warm", "warm_worker"])
    ap.add_argument("--points", required=True)
    ap.add_argument("--out")
    ap.add_argument("--iters", type=int, default=1000, help="launches timed per point; 0: shape_iters")
    ap.add_argument("--manifest")
    ap.add_argument("--frozen", help="run manifest (refit.py init): stop if a file it hashed changes")
    ap.add_argument("--foreign-wait", type=int, default=1800,
                    help="seconds to wait for foreign processes on the GPU before stopping")
    ap.add_argument("--fail-out", help="warm: write the failing points and their errors here (JSON)")
    ap.add_argument("--chunk", type=int, default=40, help="points per rocprofv3 process")
    ap.add_argument("--shard", help="i/n: time only shard i of n (PRELIM: and the repeats of shard i - 1's points)")
    ap.add_argument("--jobs", type=int, default=48, help="warm: compile processes")
    ap.add_argument("--gpus", default=os.environ.get("HIP_VISIBLE_DEVICES", "0"),
                    help="warm: comma-separated GPUs to spread the compile processes over")
    ap.add_argument("--point-timeout", type=int, default=300, help="seconds without a finished point before a kill")
    args = ap.parse_args()
    if args.mode == "worker":
        worker(args)
        return
    if args.mode == "warm_worker":
        warm_worker(args)
        return
    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
    points = points_for(args.points, shard)
    if args.mode == "warm":
        if args.frozen:
            check_frozen(args.frozen)
        warm(points, args.jobs, args.gpus.split(","), args.fail_out)
        return
    run_points(points, args.out, args.chunk, args.iters, args.point_timeout, frozen=args.frozen,
               foreign_wait=args.foreign_wait)


def run_points(points, out, chunk_size=40, iters=1000, point_timeout=300, extra=None, frozen=None, foreign_wait=1800):
    """Time the points not yet in `out` (CSV, appended chunk by chunk), in chunks of chunk_size per rocprofv3 process
    on the visible GPU, holding its check lock (gpu_lock) per chunk. A chunk starts when no foreign process is on
    the GPU (wait_for_gpu), and is dropped and timed again if one is there after it. frozen: the run manifest
    (check_frozen before every chunk). extra: columns for every row (refit.py: round)."""
    gpu = os.environ.get("HIP_VISIBLE_DEVICES", "0")
    done, chunk_id = set(), 0
    if os.path.exists(out):
        prev = list(csv.DictReader(open(out)))
        done = {point_key(r) for r in prev}
        chunk_id = 1 + max((int(r["chunk"]) for r in prev if r.get("chunk")), default=-1)
    todo = [p for p in points if p not in done]
    tag = os.path.splitext(os.path.basename(out))[0]
    total = len(todo)
    out_dir = os.path.join(TMP, f"sweep_{tag}")

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    while todo:
        if frozen:
            check_frozen(frozen)
        chunk = todo[:chunk_size]
        with gpu_lock(gpu):
            wait_for_gpu(gpu, foreign_wait, log)
            # The first point a process times comes out 3-6% fast (WORKLOG §13), so each process first times its
            # chunk's first point once more and drops it.
            res, bad = timing(chunk[:1] + chunk, out_dir, iters, point_timeout)
            res, bad = (res[1:] if res else res), (None if bad is None else max(bad - 1, 0))
            rest = todo[len(chunk):]
            if res is None:
                # Time the failing point alone; if it fails again it gets an error row. The points before it
                # go back in the queue.
                p = chunk[bad]
                res, _ = timing([p, p], out_dir, iters, point_timeout)
                res = res[1:] if res else [row(p, None, error="failed or hung twice")]
                rest = todo[:bad] + todo[bad + 1:]
            after = foreign_pids(gpu)
        if after:
            log(f"FOREIGN: pids {after} on GPU {gpu} (rocm-smi {smi_index(gpu)}) after chunk {chunk_id}; "
                f"its {len(res)} rows dropped")
            continue
        todo = rest
        for pos, r in enumerate(res):
            r.update(extra or {}, gpu=gpu, chunk=chunk_id, pos=pos)
        chunk_id += 1
        new = not os.path.exists(out)
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS if new else next(csv.reader(open(out))), extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerows(res)
        log(f"{total - len(todo)} / {total}")


if __name__ == "__main__":
    main()
