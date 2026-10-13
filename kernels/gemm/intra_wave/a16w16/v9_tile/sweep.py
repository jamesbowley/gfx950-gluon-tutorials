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
"""Timing, counters and ATT for v9_tile over a list of (tile, M, N, K) points.

    source v9_tile/env.sh            # GPU 2, llir (schedule_hint) + amdgcnas
    python v9_tile/sweep.py timing  --points A --out v9_tile/results/a_timing.csv
    python v9_tile/sweep.py counters --points A --out v9_tile/results/a_counters.csv
    python v9_tile/sweep.py att      --points A --out v9_tile/results/a_att.csv

A point is "TILE:MxNxK" (TILE as in V9_TILE, "v9" for v9_beyond_hotloop itself, "sk256" for
v26_streamk_ragged_k with every leftover tile split over all 256 programs); --points takes
a comma-separated list or the name of a preset in POINTS. All runs are bf16 with B and C rows
contiguous, as bench_ext.py runs them.

- timing: one rocprofv3 --kernel-trace process over every point. Each point is checked against
  torch, then launched 1000 times over a rotating 512 MB of inputs (cold caches); the median and
  mean of the last 100 dispatches are kept, and the median idle gap between them. From Python a
  launch takes ~135 us, so shorter kernels leave the GPU idle between launches, and an idle GPU
  boosts. --graph replays the launches from a HIP graph instead: back to back, no gap, the clock a
  long stream of these kernels settles at.
- counters: two rocprofv3 --pmc passes (counter sets below), 20 cold launches per point, median
  of launches 5-20. GRBM_GUI_ACTIVE / 8 XCDs is the kernel's length in shader cycles (it agrees
  with SQ_CYCLES / 32 SEs). MFMA busy = SQ_VALU_MFMA_BUSY_CYCLES (the MFMA work, a fixed count) /
  (kernel cycles x 1024 SIMDs): the whole kernel, idle CUs, prologue and epilogue included, unlike
  ATT's main-loop MFMA efficiency. Counter collection slows the kernel and changes the clock
  (pmc_clock_ghz), so the clock of the timing run is estimated as kernel cycles / timed duration.
- clock: no profiler. Each point runs back to back (HIP graph replays) for --seconds (after a 1 s
  idle gap) while a
  thread polls the device's hwmon (freq1_input = graphics clock, power1_input = socket power)
  every 20 ms; the last 70% of the window is averaged. Also gives the sustained TFLOPS.
- att: one rocprofv3 --att per point (CU 0, the 10th of 12 cold launches), reduced by
  scripts/process_json.py: main-loop MFMA efficiency, and the prologue / loop / epilogue shares
  of a wave's time.
"""

import argparse
import csv
import glob
import json
import os
import re
import shutil
import statistics as S
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A16 = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(A16, "..", "..", "..", ".."))
TMP = os.path.join(HERE, "tmp")

KS = (1024, 2048, 4096, 8192, 16384)
FULL_WAVE = {"256x256": (4096, 4096), "256x128": (4096, 2048), "128x256": (2048, 4096),
             "128x128": (2048, 2048), "128x128x128": (2048, 2048)}
SAME_SHAPE = [(4096, n, k) for k in (4096, 8192) for n in (1024, 2048, 3072)] + [
    (16384, 512, 8192), (1024, 2304, 16384), (2048, 2304, 16384)]
# Wave quantization against N (M = 4096): each tile over N = 256..16384, and stream-K at 256x256
# ("sk256") wherever the leftover tiles S divide the 256 CUs, so the split n = 256 / S fills them.
WQ_TILES = ("256x256", "128x256", "128x128x128")
WQ_N = range(256, 16384 + 1, 256)
WQ_SK_N = [w * 4096 + d for w in range(4) for d in (256, 512, 1024, 2048)]
TILE_RE = re.compile(r"^(tri)?(\d+)x(\d+)(?:x(\d+))?(r2)?(?:w([1248]))?(m4)?(?:e(\d))?(p)?$")


def tile_spec(tile):
    """(BM, BN, BK, regions, W, pin) as matmul_kernel.parse_spec, with v9_tile_small's default W.
    "tri..." is the Triton tl.dot reference (triton_ref.py, 4 warps unless wW)."""
    if tile in ("v9", "sk256"):
        return 256, 256, 64, 4, 4, False
    tri, bm, bn, bk, r2, w, _, _, pin = TILE_RE.match(tile).groups()
    bm, bn, bk = int(bm), int(bn), int(bk or 64)
    small = not tri and not r2 and min(bm, bn) <= 32
    w = int(w) if w else max(1, min(4, max(bm, bn) // 64)) if small else 4
    return bm, bn, bk, 2 if r2 else 4, w, bool(pin)


def is_small(tile):
    bm, bn, _, regions, _, _ = tile_spec(tile)
    return tile not in ("v9", "sk256") and not tile.startswith("tri") and regions != 2 and min(bm, bn) <= 32


def regs_ok(tile):
    """B's operand for two k-steps is (BN / W) BK / 32 registers per lane: past 8192 elements per wave
    (256 registers) the tile cannot keep one wave per SIMD (asm_small.md: 16x64x256 uses ~700)."""
    bm, bn, bk, _, w, _ = tile_spec(tile)
    return max(bm, bn) // w * bk <= 8192


def loads_ok(tile):
    """matmul_kernel.strip_load_layout: the padded A strip holds at least one 1 KB load per wave."""
    bm, bn, bk, _, w, _ = tile_spec(tile)
    return -(-min(bm, bn) // 16) * 16 >= w * (64 // min(64, bk // 8))


def pinnable(tile):
    """Whether one wave per SIMD fits: LDS holds exactly 4 / W of the tile's workgroups
    (matmul_kernel.small_pin_pads) and the registers allow it."""
    bm, bn, bk, _, w, _ = tile_spec(tile)
    km, kn = min(bm, bn), max(bm, bn)
    return 2 * (-(-km // 16) * 16 + kn) * bk * 2 <= 160 * 1024 // (4 // w) and regs_ok(tile)


# Small tiles (v9_tile_small) against the anchors, on compute-bound shapes (~1000 FLOP/B at HBM):
# each small tile with natural occupancy and pinned to one wave per SIMD ("p"), where LDS allows.
SMALL_CB_TILES = ("32x256x64", "32x128x128", "32x64x128", "32x32x128", "16x256x128", "16x256x64w2",
                  "16x128x128", "16x64x128", "16x64x64", "16x32x128", "16x16x128", "16x16x256",
                  "8x256x128", "4x256x128", "2x256x128", "1x256x128", "256x16x128")
SMALL_ANCHORS = ("256x256", "128x128x128", "64x64", "64x64x128")
# Memory-bound: M = BM rows, the weights B (N x K) streamed once.
SMALL_MB_M = (1, 4, 16, 32)
SMALL_MB_N = (4096, 8192, 16384)
SMALL_MB_BN = (16, 32, 64, 128, 256)
# Alternatives to v9_tile_small's choices (WORKLOG §14.9), and the Triton tl.dot reference ("tri", run
# stock: no LLVM scheduler, no amdgcnas; SMALL_REF_*). m4: the 4x64 MFMA; wW above BN / 16: replicated
# warps; eE: waves_per_eu = E instead of the LDS pin.
ALT_CB = "2048x4096x8192"
ALT_CB_SMALL_N = "2048x256x8192"
ALT_WPE_TILES = ("16x64x64", "16x32x128", "16x16x128", "32x256x64")
ALT_POINTS = {
    "SMALL_ALT_CB": [f"{bm}x256x128{s}:{ALT_CB}" for bm in (1, 2, 4, 8) for s in ("", "m4")]
    + [f"256x{bn}x128{s}:{ALT_CB_SMALL_N}" for bn in (4, 8) for s in ("", "m4")]
    + [f"16x16x128{s}:{ALT_CB}" for s in ("", "p", "w4")]
    + [f"{t}{s}:{ALT_CB}" for t in ALT_WPE_TILES for s in ("", "p", "e1", "e2")
       if not (t == "16x16x128" and s in ("", "p"))]
    + [f"16x256x128:{ALT_CB}", f"16x64x128:{ALT_CB}"],
    "SMALL_REF_CB": [f"tri{bm}x256x128:{ALT_CB}" for bm in (1, 2, 4, 8, 16)]
    + [f"tri256x{bn}x128:{ALT_CB_SMALL_N}" for bn in (4, 8)]
    + [f"tri16x16x128{w}:{ALT_CB}" for w in ("w1", "")] + [f"tri16x64x128w1:{ALT_CB}"],
    "SMALL_ALT_MB": [f"{m}x{bn}x128{s}:{m}x{n}x8192" for m in (1, 4, 8) for n in (8192, 16384) for bn in (64, 256)
                     for s in ("", "m4")]
    + [f"{bn}x{m}x128{s}:{n}x{m}x8192" for m in (4, 8) for n in (8192, 16384) for bn in (64, 256) for s in ("", "m4")],
    "SMALL_REF_MB": [f"tri{m}x{bn}x128{'w1' if bn == 64 else ''}:{m}x{n}x8192" for m in (1, 4, 8) for n in (8192, 16384)
                     for bn in (64, 256)]
    + [f"tri{bn}x{m}x128{'w1' if bn == 64 else ''}:{n}x{m}x8192" for m in (4, 8) for n in (8192, 16384)
       for bn in (64, 256)],
}
POINTS = {
    "A": [f"{t}:{m}x{n}x{k}" for t in ("256x256", "256x128", "128x256") for k in KS
          for m, n in [FULL_WAVE[t]]],
    "SMALL_CB": [f"{t}{p}:2048x4096x{k}" for k in (4096, 8192) for t in SMALL_CB_TILES
                 for p in ("", "p") if not p or pinnable(t)]
    + [f"{t}:2048x4096x{k}" for k in (4096, 8192) for t in SMALL_ANCHORS],
    "SMALL_MB": [f"{m}x{bn}x128{p}:{m}x{n}x8192" for m in SMALL_MB_M for n in SMALL_MB_N for bn in SMALL_MB_BN
                 for p in ("", "p") if not p or pinnable(f"{m}x{bn}x128")]
    + [f"16x{bn}x{bk}:16x{n}x8192" for n in SMALL_MB_N for bn in SMALL_MB_BN for bk in (64, 256)
       if regs_ok(f"16x{bn}x{bk}") and loads_ok(f"16x{bn}x{bk}")],
    **ALT_POINTS,
    # Run with --rotating-mb 0 (inputs warm in the MALL): the window ceiling with a shorter round trip.
    "SMALL_WARM": [f"{t}:2048x4096x8192" for t in ("256x256", "64x64x128", "32x32x128p", "16x256x128", "16x64x128p",
                                                   "16x16x128p", "1x256x128")],
    "A128": [f"{t}:2048x2048x{k}" for t in ("128x128", "128x128r2", "128x128x128") for k in KS],
    "B": [f"{t}:{m}x{n}x{k}" for m, n, k in SAME_SHAPE for t in ("256x256", "256x128", "128x256")],
    "B128": [f"{t}:{m}x{n}x{k}" for m, n, k in SAME_SHAPE for t in ("128x128", "128x128r2", "128x128x128")],
    **{f"WQ{k}": [f"{t}:4096x{n}x{k}" for t in WQ_TILES for n in WQ_N] for k in (4096, 8192)},
    **{f"WQSK{k}": [f"sk256:4096x{n}x{k}" for n in WQ_SK_N] for k in (4096, 8192)},
}
# sk256 runs v26 with every leftover tile split over all 256 programs (the default capped policy
# would use at most 128 of them for small S); the ending is the policy's (rs for n >= 8, else owner).
SK_ENV = {"STREAMK_POLICY": "one_tile", "STREAMK_NUM_PROGRAMS": "256"}
PEAK_TFLOPS_PER_GHZ = 256 * 4096 / 1e3  # 256 CUs x 4096 FLOP per clock
COUNTER_SETS = [
    ["GRBM_GUI_ACTIVE", "SQ_VALU_MFMA_BUSY_CYCLES", "SQ_BUSY_CU_CYCLES"],
    ["TCC_HIT_sum", "TCC_MISS_sum", "TCC_EA0_RDREQ_sum", "TCC_EA0_RDREQ_DRAM_sum"],
    ["TCC_EA0_RDREQ_32B_sum", "TCC_EA0_RDREQ_64B_sum", "TCC_EA0_RDREQ_128B_sum"],
    ["MeanOccupancyPerActiveCU"],
    ["GRBM_GUI_ACTIVE", "SQ_LDS_BANK_CONFLICT", "SQ_LDS_IDX_ACTIVE", "SQ_INSTS_LDS"],
]
DEFAULT_COUNTER_SETS = "0,1,2,3"  # --sets; 4 (LDS bank conflicts) on request
GRAPH_LEN = 100
ROTATING_MB = 512  # --rotating-mb; 0 reuses one set of inputs (warm caches)
NUM_SIMDS = 1024
NUM_XCDS = 8  # GRBM_GUI_ACTIVE comes back summed over the 8 XCDs


def parse_point(p):
    tile, shape = p.split(":")
    M, N, K = (int(x) for x in shape.split("x"))
    return tile, M, N, K


KERNELS = ("v9_tile", "v9_tile_2r", "v9_tile_small", "v9_beyond_hotloop", "v26_streamk_ragged_k", "triton_ref_matmul")


def kernel_name(tile):
    if tile == "sk256":
        return "v26_streamk_ragged_k"
    if tile == "v9":
        return "v9_beyond_hotloop"
    if tile.startswith("tri"):
        return "triton_ref_matmul"
    return "v9_tile_small" if is_small(tile) else "v9_tile_2r" if tile_spec(tile)[3] == 2 else "v9_tile"


def tile_dims(tile):
    return tile_spec(tile)[:3]


def runner(tile):
    """The point's matmul(a, b, c)."""
    import importlib

    if tile == "v9":
        return importlib.import_module("v9_beyond_hotloop.matmul_kernel").matmul
    if tile == "sk256":
        os.environ.update(SK_ENV)
        return importlib.import_module("v26_streamk_ragged_k.matmul_kernel").matmul
    if tile.startswith("tri"):
        ref = importlib.import_module("v9_tile.triton_ref")
        return lambda a, b, c: ref.matmul(a, b, c, tile)
    mk = importlib.import_module("v9_tile.matmul_kernel")
    return lambda a, b, c: mk.matmul(a, b, c, tile=tile)


def busy_cus(tile, tiles):
    """Average busy CUs: stream-K keeps all 256 busy; a non-persistent launch runs its tiles in
    ceil(tiles / 256) waves."""
    return 256 if tile == "sk256" else tiles / -(-tiles // 256)


# ---------------------------------------------------------------------------------------------
# worker: runs inside rocprofv3
# ---------------------------------------------------------------------------------------------
def worker(args):
    sys.path.insert(0, A16)
    import bench_ext as bench  # RTLD_GLOBAL, amdgcnas hook, utils path
    import torch

    manifest = []
    for p in args.points:
        tile, M, N, K = parse_point(p)
        run = runner(tile)
        dt = torch.bfloat16
        a = torch.rand((M, K), device="cuda", dtype=dt) - 0.5
        b = bench.make_b(N, K, dt, False, lambda t: t.uniform_(-0.5, 0.5))
        c = bench.make_c(M, N, dt, False)
        run(a, b, c)
        ref = (a.float() @ b.float()).to(dt)
        correct = bool(torch.allclose(c, ref, atol=1e-1, rtol=1e-2))
        del a, b, c, ref
        a_l, b_l, c_l, copies = bench.gen_rotating_tensors(M, N, K, dt, args.rotating_mb)
        if args.graph:
            # One HIP graph of GRAPH_LEN launches over the rotating copies, replayed: the launches
            # run back to back, with none of the host's ~135 us per-launch gap.
            for i in range(copies):
                run(a_l[i], b_l[i], c_l[i])
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for i in range(GRAPH_LEN):
                    run(a_l[i % copies], b_l[i % copies], c_l[i % copies])
            for _ in range(args.iters // GRAPH_LEN):
                g.replay()
            torch.cuda.synchronize()
            dispatches = 1 + copies + (args.iters // GRAPH_LEN) * GRAPH_LEN
            del g
        else:
            for i in range(args.iters):
                run(a_l[i % copies], b_l[i % copies], c_l[i % copies])
            torch.cuda.synchronize()
            dispatches = 1 + args.iters
        del a_l, b_l, c_l
        torch.cuda.empty_cache()
        manifest.append({"point": p, "dispatches": dispatches, "correct": correct})
        print(f"{p} copies={copies} correct={correct}", flush=True)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=1)


def run_worker(prof_args, points, iters, out_dir, extra_env=None, graph=False):
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    manifest = os.path.join(out_dir, "manifest.json")
    regex = "|".join(sorted({kernel_name(parse_point(p)[0]) for p in points}))
    cmd = ["rocprofv3", *prof_args, "--kernel-include-regex", regex, "-d", out_dir, "-o", "run",
           "--output-format", "csv", "--", sys.executable, os.path.abspath(__file__), "worker",
           "--iters", str(iters), "--manifest", manifest, "--points", ",".join(points),
           "--rotating-mb", str(ROTATING_MB)] + (["--graph"] if graph else [])
    p = subprocess.run(cmd, cwd=A16, env={**os.environ, **(extra_env or {})}, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-3000:], p.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"rocprofv3 failed ({p.returncode})")
    return json.load(open(manifest))


def chunks(rows, manifest):
    """Split dispatch-ordered rows into per-point lists, checking the counts."""
    expected = sum(m["dispatches"] for m in manifest)
    assert len(rows) == expected, f"{len(rows)} dispatches in the trace, expected {expected}"
    out, pos = [], 0
    for m in manifest:
        out.append(rows[pos : pos + m["dispatches"]])
        pos += m["dispatches"]
    return out


def read_csv(out_dir, suffix):
    """The GEMM kernels' rows (rocprofv3's include regex does not filter the torch kernels out)."""
    files = glob.glob(os.path.join(out_dir, "**", f"*{suffix}"), recursive=True)
    assert len(files) == 1, f"expected one *{suffix} in {out_dir}, found {files}"
    return [r for r in csv.DictReader(open(files[0])) if r["Kernel_Name"] in KERNELS]


def base_row(p, correct):
    tile, M, N, K = parse_point(p)
    BM, BN, BK = tile_dims(tile)
    return {"tile": tile, "M": M, "N": N, "K": K, "tiles": -(-M // BM) * -(-N // BN), "correct": correct}


def unique_bytes(r):
    """Bytes of A, B and C (bf16), each touched once: the memory-bound roof's numerator."""
    return 2 * (r["M"] * r["K"] + r["K"] * r["N"] + r["M"] * r["N"])


# ---------------------------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------------------------
def timing(args):
    out_dir = os.path.join(TMP, "sweep_timing" + ("_graph" if args.graph else ""))
    manifest = run_worker(["--kernel-trace"], args.points, 1000, out_dir, graph=args.graph)
    rows = read_csv(out_dir, "kernel_trace.csv")
    rows.sort(key=lambda r: int(r["Dispatch_Id"]))
    res = []
    for m, chunk in zip(manifest, chunks(rows, manifest)):
        tail = chunk[-100:]
        us = [(int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3 for r in tail]
        gaps = [(int(b["Start_Timestamp"]) - int(a["End_Timestamp"])) / 1e3 for a, b in zip(tail, tail[1:])]
        r = base_row(m["point"], m["correct"])
        med = S.median(us)
        r.update({"us_median": round(med, 3), "us_mean": round(S.mean(us), 3),
                  "tflops": round(2 * r["M"] * r["N"] * r["K"] / (med * 1e-6) / 1e12, 1),
                  "tbps": round(unique_bytes(r) / (med * 1e-6) / 1e12, 3),
                  "gap_us": round(S.median(gaps), 1)})
        res.append(r)
    return res


def counters(args):
    res = None
    for i in (int(s) for s in args.sets.split(",")):
        cset = COUNTER_SETS[i]
        out_dir = os.path.join(TMP, f"sweep_counters_{i}")
        manifest = run_worker(["--kernel-trace", "--pmc", *cset], args.points, 20, out_dir)
        trace = sorted(read_csv(out_dir, "kernel_trace.csv"), key=lambda r: int(r["Dispatch_Id"]))
        dur = {int(r["Dispatch_Id"]): int(r["End_Timestamp"]) - int(r["Start_Timestamp"]) for r in trace}
        vals = {}
        for r in read_csv(out_dir, "counter_collection.csv"):
            vals.setdefault(int(r["Dispatch_Id"]), {})
            vals[int(r["Dispatch_Id"])][r["Counter_Name"]] = vals[int(r["Dispatch_Id"])].get(
                r["Counter_Name"], 0.0) + float(r["Counter_Value"])
        ids = sorted(vals)
        per_point = chunks(ids, manifest)
        if res is None:
            res = [base_row(m["point"], m["correct"]) for m in manifest]
        for r, disp in zip(res, per_point):
            disp = disp[5:]
            med = lambda f: S.median(f(vals[d], dur[d]) for d in disp)  # noqa: E731
            if i == 0:
                r["kcycles"] = round(med(lambda v, t: v["GRBM_GUI_ACTIVE"] / NUM_XCDS / 1e3), 1)
                r["mfma_busy"] = round(med(lambda v, t: v["SQ_VALU_MFMA_BUSY_CYCLES"] /
                                           (v["GRBM_GUI_ACTIVE"] / NUM_XCDS * NUM_SIMDS)), 4)
                r["pmc_us"] = round(med(lambda v, t: t / 1e3), 2)
                r["pmc_clock_ghz"] = round(med(lambda v, t: v["GRBM_GUI_ACTIVE"] / NUM_XCDS / t), 3)
            elif i == 1:
                r["l2_hit"] = round(med(lambda v, t: v["TCC_HIT_sum"] / (v["TCC_HIT_sum"] + v["TCC_MISS_sum"])), 4)
                r["l2_req_M"] = round(med(lambda v, t: (v["TCC_HIT_sum"] + v["TCC_MISS_sum"]) / 1e6), 3)
                r["ea_rdreq_M"] = round(med(lambda v, t: v["TCC_EA0_RDREQ_sum"] / 1e6), 3)
                r["ea_rdreq_dram_M"] = round(med(lambda v, t: v["TCC_EA0_RDREQ_DRAM_sum"] / 1e6), 3)
            elif i == 2:
                # Bytes the L2 read from beyond itself (MALL or HBM), by request size, against the
                # bytes of A, B and C: over 1 means re-reads, under 1 data the L2 already held.
                ea = med(lambda v, t: 32 * v["TCC_EA0_RDREQ_32B_sum"] + 64 * v["TCC_EA0_RDREQ_64B_sum"]
                         + 128 * v["TCC_EA0_RDREQ_128B_sum"])
                r["ea_rd_MB"] = round(ea / 1e6, 2)
                r["ea_rd_over_unique"] = round(ea / unique_bytes(r), 3)
            elif i == 3:
                # Reads 1.0 at one wave per SIMD (the pinned runs, 4 waves per CU): waves per SIMD.
                r["waves_per_simd"] = round(med(lambda v, t: v["MeanOccupancyPerActiveCU"]), 2)
            else:
                # LDS bank conflicts, summed over the CUs: conflict cycles per conflict-free LDS cycle
                # (rocprof's LDS_BANK_CONFLICT_ACCESS), and as a share of every CU's kernel cycles.
                cu_cycles = lambda v: v["GRBM_GUI_ACTIVE"] / NUM_XCDS * 256  # noqa: E731
                r["lds_conflict_per_access"] = round(med(lambda v, t: v["SQ_LDS_BANK_CONFLICT"] / max(
                    1.0, v["SQ_LDS_IDX_ACTIVE"] - v["SQ_LDS_BANK_CONFLICT"])), 4)
                r["lds_conflict_pct"] = round(med(lambda v, t: v["SQ_LDS_BANK_CONFLICT"] / cu_cycles(v)), 4)
                r["lds_active_pct"] = round(med(lambda v, t: v["SQ_LDS_IDX_ACTIVE"] / cu_cycles(v)), 4)
                r["lds_conflict_per_inst"] = round(med(lambda v, t: v["SQ_LDS_BANK_CONFLICT"] / max(
                    1.0, v["SQ_INSTS_LDS"])), 3)
    return res


def clock_worker(args):
    """Each point back to back for args.seconds over rotating inputs; wall times to the manifest."""
    import time
    sys.path.insert(0, A16)
    import bench_ext as bench  # noqa: F401
    import torch

    out = []
    for p in args.points:
        tile, M, N, K = parse_point(p)
        run = runner(tile)
        a_l, b_l, c_l, copies = bench.gen_rotating_tensors(M, N, K, torch.bfloat16, args.rotating_mb)
        for i in range(copies):
            run(a_l[i], b_l[i], c_l[i])
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(GRAPH_LEN):
                run(a_l[i % copies], b_l[i % copies], c_l[i % copies])
        torch.cuda.synchronize()
        time.sleep(1.0)  # idle gap so every point starts from the same state
        t0, n = time.time(), 0
        while time.time() - t0 < args.seconds:
            for _ in range(5):
                g.replay()
            n += 5 * GRAPH_LEN
            torch.cuda.synchronize()
        t1 = time.time()
        out.append({"point": p, "t0": t0, "t1": t1, "launches": n})
        del a_l, b_l, c_l, g
        torch.cuda.empty_cache()
    json.dump(out, open(args.manifest, "w"), indent=1)


def hwmon_dir():
    """hwmon of the device HIP_VISIBLE_DEVICES selects (matched by PCI bus, single-GPU runs only)."""
    import torch
    bus = torch.cuda.get_device_properties(0).pci_bus_id
    for card in glob.glob("/sys/class/drm/card*/device"):
        if os.path.basename(os.path.realpath(card)).endswith(f":{bus:02x}:00.0"):
            return glob.glob(os.path.join(card, "hwmon", "hwmon*"))[0]
    raise SystemExit(f"no hwmon for PCI bus {bus:#x}")


def clock(args):
    import threading
    import time
    hw = hwmon_dir()
    samples, stop = [], threading.Event()

    def poll():
        while not stop.is_set():
            f = int(open(os.path.join(hw, "freq1_input")).read()) / 1e9
            w = int(open(os.path.join(hw, "power1_input")).read()) / 1e6
            samples.append((time.time(), f, w))
            time.sleep(0.02)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    manifest = os.path.join(TMP, "sweep_clock.json")
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "clock_worker", "--points", ",".join(args.points),
                        "--manifest", manifest, "--seconds", str(args.seconds), "--rotating-mb", str(ROTATING_MB)],
                       cwd=A16, capture_output=True, text=True)
    stop.set()
    th.join()
    if p.returncode:
        print(p.stderr[-3000:], file=sys.stderr)
        raise SystemExit("clock worker failed")
    res = []
    for m in json.load(open(manifest)):
        r = base_row(m["point"], None)
        r.pop("correct")
        lo = m["t0"] + 0.3 * (m["t1"] - m["t0"])
        win = [s for s in samples if lo <= s[0] <= m["t1"]]
        tflops = 2 * r["M"] * r["N"] * r["K"] * m["launches"] / (m["t1"] - m["t0"]) / 1e12
        ghz = S.mean(s[1] for s in win)
        util = tflops / (PEAK_TFLOPS_PER_GHZ * ghz)
        r.update({"sustained_tflops": round(tflops, 1), "sclk_ghz": round(ghz, 3),
                  "power_w": round(S.mean(s[2] for s in win)), "samples": len(win),
                  "mfma_util": round(util, 4),
                  "mfma_util_busy": round(util * 256 / busy_cus(r["tile"], r["tiles"]), 4)})
        res.append(r)
    return res


ATT_JOB = {"jobs": [{"kernel_include_regex": "", "kernel_exclude_regex": "", "kernel_iteration_range": "[10]",
                     "advanced_thread_trace": True, "att_target_cu": 0, "att_shader_engine_mask": "0xF",
                     "att_simd_select": "0xF", "att_buffer_size": "0x60000000"}]}


def att(args):
    res = []
    env = {**os.environ, "ROCPROF_ATT_LIBRARY_PATH": os.environ.get("ROCPROF_ATT_LIBRARY_PATH", "/opt/rocm/lib/")}
    for p in args.points:
        tile, M, N, K = parse_point(p)
        out_dir = os.path.join(TMP, "sweep_att", p.replace(":", "_"))
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
        job = json.loads(json.dumps(ATT_JOB))
        job["jobs"][0]["kernel_include_regex"] = kernel_name(tile)
        cfg = os.path.join(out_dir, "att.json")
        json.dump(job, open(cfg, "w"))
        manifest = os.path.join(out_dir, "manifest.json")
        cmd = ["rocprofv3", "--att", "-i", cfg, "-d", out_dir, "--", sys.executable, os.path.abspath(__file__),
               "worker", "--iters", "11", "--manifest", manifest, "--points", p, "--rotating-mb", str(ROTATING_MB)]
        pr = subprocess.run(cmd, cwd=A16, env=env, capture_output=True, text=True)
        row = base_row(p, None)
        ui = glob.glob(os.path.join(out_dir, "**", "ui_*"), recursive=True)
        if pr.returncode or not ui:
            print(f"{p}: ATT failed\n{pr.stderr[-2000:]}", file=sys.stderr)
            res.append(row)
            continue
        row["correct"] = json.load(open(manifest))[0]["correct"]
        pj = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "process_json.py"), ui[0]],
                            capture_output=True, text=True)
        try:
            j = json.loads(pj.stdout)
        except json.JSONDecodeError:
            print(f"{p}: process_json failed\n{pj.stderr[-2000:]}", file=sys.stderr)
            res.append(row)
            continue
        if j["mfma efficiency"] == "N/A":  # process_json found no loop (e.g. LLVM rotated it)
            print(f"{p}: process_json found no loop", flush=True)
            res.append(row)
            continue
        row.update({"att_mfma_eff": float(j["mfma efficiency"].rstrip("%")) / 100,
                    "mfma_in_loop": j["mfma_count_in_loop"],
                    "iter_cycles": round(j["average_iteration_duration"], 1),
                    "pro_ratio": j["pro_ratio"], "loop_ratio": j["loop_ratio"], "epi_ratio": j["epi_ratio"],
                    "prologue_cycles": round(j["average_prologue_duration"]),
                    "epilogue_cycles": round(j["average_epilogue_duration"])})
        print(f"{p}: {j['mfma efficiency']}", flush=True)
        res.append(row)
    return res


def main():
    global ROTATING_MB, TMP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=["timing", "counters", "att", "clock", "worker", "clock_worker"])
    ap.add_argument("--points", required=True)
    ap.add_argument("--out")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--rotating-mb", type=int, default=512)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--graph", action="store_true", help="timing: launch from a HIP graph, back to back")
    ap.add_argument("--sets", default=DEFAULT_COUNTER_SETS,
                    help="counters: indices into COUNTER_SETS (4: LDS bank conflicts)")
    ap.add_argument("--manifest")
    ap.add_argument("--tmp", default=TMP, help="scratch directory (one per concurrent sweep)")
    args = ap.parse_args()
    args.points = POINTS.get(args.points) or [p for p in args.points.split(",") if p]
    ROTATING_MB = args.rotating_mb
    TMP = args.tmp
    if args.mode in ("worker", "clock_worker"):
        (worker if args.mode == "worker" else clock_worker)(args)
        return
    res = {"timing": timing, "counters": counters, "att": att, "clock": clock}[args.mode](args)
    keys = list(dict.fromkeys(k for r in res for k in r))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(res)
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for r in res:
        print("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")


if __name__ == "__main__":
    main()
