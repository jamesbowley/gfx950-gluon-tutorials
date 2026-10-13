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
"""Sweep driver for the HBM bandwidth vs CU/XCD placement experiments.

    HIP_VISIBLE_DEVICES=0 python bench.py --sweep check A B C D E F

Each sweep writes results/<sweep>.csv. See README.md for what each sweep measures.
"""

import argparse
import csv
import itertools
import os
import statistics
import subprocess
import time
from dataclasses import replace

import triton
import torch

from kernel import (BWConfig, Buffers, SplitKConfig, prepare, prepare_splitk, read_records, NUM_XCDS, NUM_CUS)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
MB = 1 << 20
GB = 1 << 30

# Default "memory-bound kernel" shape: 8 waves, 4 x 16 B loads in flight per thread (32 KB per workgroup).
DEFAULT = dict(num_warps=8, unroll=4)
W_LIST = list(range(1, 17)) + list(range(20, 65, 4)) + list(range(72, 257, 8))


# ----------------------------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------------------------
class Flusher:
    """Evicts L2 and MALL between timed runs by streaming 1 GB of reads (clean lines, no write-back)."""

    def __init__(self):
        self.bufs = Buffers(GB // 4, dst_elems=1)
        cfg = BWConfig(op="read", W=NUM_CUS, n_iter=(GB // 4) // (NUM_CUS * BWConfig(**DEFAULT).tile_elems),
                       record=False, **DEFAULT)
        self.launch = prepare(cfg, self.bufs)

    def __call__(self):
        self.launch()


def time_launch(launch, iters, flush, cold=True, warmup=2, pre=None):
    pre = pre or getattr(launch, "pre", None) or (lambda: None)
    for _ in range(warmup):
        pre()
        launch()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if cold:
            flush()
        pre()
        starts[i].record()
        launch()
        ends[i].record()
    torch.cuda.synchronize()
    ts = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return statistics.median(ts), min(ts), max(ts)


def analyse_records(cfg, bufs):
    """Placement and in-kernel timing from the last launch's per-workgroup records."""
    r = read_records(cfg, bufs)
    act = r["active"]
    n_act = int(act.sum())
    if n_act == 0:
        return dict(n_active=0)
    keys = r["cu_key"][act]
    xcc = r["xcc"][act]
    t0 = r["t0"][act].double()
    t1 = r["t1"][act].double()
    span_us = (t1.max() - t0.min()).item() / 100.0
    dt = (t1 - t0).clamp(min=1)
    clk = ((r["c1"][act] - r["c0"][act]).double() / dt * 100.0)
    per_xcd = torch.bincount(xcc, minlength=NUM_XCDS).tolist()
    _, cu_counts = torch.unique(keys, return_counts=True)
    return dict(
        n_active=n_act,
        n_xcds=len(set(xcc.tolist())),
        wgs_per_xcd="/".join(str(c) for c in per_xcd),
        n_cus=len(cu_counts),
        max_wg_per_cu=int(cu_counts.max()),
        kernel_span_us=round(span_us, 2),
        start_skew_us=round((t0.max() - t0.min()).item() / 100.0, 2),
        wg_time_mean_us=round(dt.mean().item() / 100.0, 2),
        wg_time_max_us=round(dt.max().item() / 100.0, 2),
        sclk_mhz=round(clk.median().item(), 0),
    )


def run_cfg(cfg, bufs, flush, iters, cold=True, **extra):
    launch = prepare(cfg, bufs)
    med, lo, hi = time_launch(launch, iters, flush, cold=cold)
    row = dict(extra)
    row.update(op=cfg.op, mode=cfg.mode, pattern=cfg.pattern, W=cfg.W, num_warps=cfg.num_warps, unroll=cfg.unroll,
               wpc=cfg.wpc, n_iter=cfg.n_iter, passes=cfg.passes, footprint_bytes=cfg.footprint_elems * 4,
               bytes=cfg.bytes_moved, cold=int(cold), time_ms=round(med, 4),
               time_min_ms=round(lo, 4), time_max_ms=round(hi, 4), gbps=round(cfg.bytes_moved / (med * 1e-3) / 1e9,
                                                                                1))
    if cfg.record:
        rec = analyse_records(cfg, bufs)
        if rec.get("kernel_span_us"):
            rec["gbps_in_kernel"] = round(cfg.bytes_moved / (rec["kernel_span_us"] * 1e-6) / 1e9, 1)
            rec["overhead_us"] = round(med * 1e3 - rec["kernel_span_us"], 2)
        row.update(rec)
    return row


def with_passes(cfg, passes):
    return replace(cfg, n_iter=cfg.wrap * passes, passes=passes)


def pick_passes(cfg, bufs, target_ms=0.25, probe_passes=8):
    """Smallest P such that P passes over cfg's footprint take about target_ms (in-kernel, warm)."""
    probe = with_passes(cfg, probe_passes)
    launch = prepare(probe, bufs)
    for _ in range(2):
        launch.pre()
        launch()
    torch.cuda.synchronize()
    span_us = analyse_records(probe, bufs).get("kernel_span_us") or 1.0
    per_pass_ms = span_us * 1e-3 / probe_passes
    return max(2, int(target_ms / per_pass_ms + 0.999))


def run_passes(cfg, bufs, flush, iters, passes=None, **extra):
    """Steady-state bandwidth over cfg's footprint from the difference between P and 4P passes.

    Subtracting the P-pass run removes launch overhead, ramp-up and the cold first pass. P is chosen so that the
    4P run lasts about 1 ms unless given.
    """
    P = passes or pick_passes(cfg, bufs)
    lo_cfg, hi_cfg = with_passes(cfg, P), with_passes(cfg, 4 * P)
    lo = run_cfg(lo_cfg, bufs, flush, iters)
    hi = run_cfg(hi_cfg, bufs, flush, iters)
    d_bytes = hi_cfg.bytes_moved - lo_cfg.bytes_moved
    row = dict(extra)
    row.update({k: hi[k] for k in hi if k not in ("time_ms", "time_min_ms", "time_max_ms", "gbps", "bytes")})
    row.update(footprint_per_wg_kb=cfg.wrap * cfg.tile_elems * 4 // 1024, P=P, time_P_ms=lo["time_ms"],
               time_4P_ms=hi["time_ms"], gbps_4P=hi["gbps"],
               slope_gbps=round(d_bytes / ((hi["time_ms"] - lo["time_ms"]) * 1e-3) / 1e9, 1))
    if lo.get("kernel_span_us") and hi.get("kernel_span_us"):
        row["slope_gbps_kernel"] = round(d_bytes / ((hi["kernel_span_us"] - lo["kernel_span_us"]) * 1e-6) / 1e9, 1)
    return row


def footprint_cfg(op, per_wg_kb, num_warps=4, unroll=4, **kw):
    """Config whose workers each own per_wg_kb of data (passes filled in later by run_passes)."""
    cfg = BWConfig(op=op, num_warps=num_warps, unroll=unroll, **kw)
    wrap = max(1, per_wg_kb * 1024 // (cfg.tile_elems * 4))
    return replace(cfg, n_iter=wrap, passes=1)


class CSVOut:

    def __init__(self, name):
        os.makedirs(RESULTS, exist_ok=True)
        self.path = os.path.join(RESULTS, f"{name}.csv")
        self.rows = []

    def add(self, row, echo=True):
        self.rows.append(row)
        if echo:
            keys = ["sweep_tag", "op", "mode", "W", "num_warps", "unroll", "wpc", "bytes", "time_ms", "gbps",
                    "gbps_in_kernel", "n_xcds", "n_cus", "max_wg_per_cu", "sclk_mhz"]
            print("  ".join(f"{k}={row[k]}" for k in keys if k in row), flush=True)

    def save(self):
        fields = []
        for r in self.rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(self.rows)
        print(f"wrote {self.path} ({len(self.rows)} rows)")


def log_clocks(name):
    gpu = os.environ.get("HIP_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        txt = subprocess.run(["amd-smi", "metric", "-g", gpu, "-c", "-p"], capture_output=True, text=True,
                             timeout=30).stdout
    except Exception as e:  # noqa: BLE001
        txt = f"amd-smi failed: {e}"
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"{name}_clocks.txt"), "a") as f:
        f.write(f"=== {time.ctime()} (physical GPU {gpu})\n{txt}\n")


def n_iter_for(bytes_per_op_buffer, W, tile_elems):
    return max(1, bytes_per_op_buffer // (W * tile_elems * 4))


# ----------------------------------------------------------------------------------------------
# Sweeps
# ----------------------------------------------------------------------------------------------
def discover_cus(bufs):
    """CU keys per XCD from a record-only launch (every workgroup exits early)."""
    cfg = BWConfig(mode="rr", W=0, record=True)
    prepare(cfg, bufs)()
    torch.cuda.synchronize()
    r = read_records(cfg, bufs)
    per_xcd = {x: [] for x in range(NUM_XCDS)}
    for x, k in zip(r["xcc"].tolist(), r["cu_key"].tolist()):
        if k not in per_xcd[x]:
            per_xcd[x].append(k)
    return per_xcd, r


def pick_cus(per_xcd, n_per_xcd):
    """n CUs per XCD, spread round-robin over shader engines."""
    out = []
    for x in range(NUM_XCDS):
        by_se = {}
        for k in sorted(per_xcd[x]):
            by_se.setdefault((k % 256) // 32, []).append(k)
        order = [k for grp in itertools.zip_longest(*by_se.values()) for k in grp if k is not None]
        out += order[:n_per_xcd]
    return out


def sweep_check(bufs, flush, args):
    """Placement validation: pid -> XCD mapping, distinct CUs per mode, WPC packing, HWSEL."""
    out = CSVOut("check")
    per_xcd, r = discover_cus(bufs)
    xcc_ok = bool((r["xcc"] == r["pid"] % NUM_XCDS).all())
    print(f"pid % 8 == XCC_ID for all {len(r['pid'])} pids: {xcc_ok}")
    print("CUs per XCD:", {x: len(v) for x, v in per_xcd.items()})
    n_iter = n_iter_for(256 * MB, 32, BWConfig(**DEFAULT).tile_elems)
    cases = [
        ("rr32", BWConfig(mode="rr", W=32)),
        ("rr32_plain", BWConfig(mode="rr", W=32, plain_grid=True)),
        ("packed32", BWConfig(mode="packed", W=32)),
        ("packed64", BWConfig(mode="packed", W=64)),
        ("packed32_wpc2", BWConfig(mode="packed", W=32, wpc=2)),
        ("subset_x1x5", BWConfig(mode="subset", W=64, xcd_mask=(1, 5), per_xcd=32)),
        ("rr256", BWConfig(mode="rr", W=256)),
        ("rr512_wpc2", BWConfig(mode="rr", W=512, wpc=2)),
        ("hwsel32cu_cap1", BWConfig(mode="hwsel", W=32, hwsel_caps={k: 1 for k in pick_cus(per_xcd, 4)})),
        ("hwsel32cu_cap2", BWConfig(mode="hwsel", W=64, wpc=2, hwsel_caps={k: 2 for k in pick_cus(per_xcd, 4)})),
        ("hwsel64cu_cap1", BWConfig(mode="hwsel", W=64, hwsel_caps={k: 1 for k in pick_cus(per_xcd, 8)})),
    ]
    for tag, cfg in cases:
        cfg = replace(cfg, n_iter=n_iter, **DEFAULT)
        row = run_cfg(cfg, bufs, flush, iters=3, sweep_tag=tag)
        out.add(row)
    out.rows.insert(0, dict(sweep_tag="pid_mod8_eq_xcc", op="", mode="", W=len(r["pid"]), n_xcds=int(xcc_ok)))
    out.save()


def sweep_T(bufs, flush, args):
    """Quick tuning: per-CU and full-chip bandwidth vs waves and loads in flight."""
    out = CSVOut("T")
    for op, W, nw, u in itertools.product(["read", "copy"], [1, 32, 256], [4, 8, 16], [2, 4, 8]):
        cfg = BWConfig(op=op, mode="rr", W=W, num_warps=nw, unroll=u)
        cfg = replace(cfg, n_iter=n_iter_for(32 * MB, 1, cfg.tile_elems))
        out.add(run_cfg(cfg, bufs, flush, args.iters, sweep_tag="tune"))
    out.save()


def sweep_A(bufs, flush, args):
    """CU sweep, round-robin vs XCD-packed, weak (32 MB per worker) and strong (2 GB total) scaling."""
    out = CSVOut("A")
    ws = list(range(1, 257)) if args.full else W_LIST
    tile = BWConfig(**DEFAULT).tile_elems
    for scaling in ["weak", "strong"]:
        for op in ["read", "write", "copy"]:
            for mode in ["rr", "packed"]:
                for W in ws:
                    n_iter = n_iter_for(32 * MB, 1, tile) if scaling == "weak" else n_iter_for(2 * GB, W, tile)
                    cfg = BWConfig(op=op, mode=mode, W=W, n_iter=n_iter, **DEFAULT)
                    out.add(run_cfg(cfg, bufs, flush, args.iters, sweep_tag=scaling, scaling=scaling))
        # plain grid of W workgroups vs 256-with-early-exit, to confirm the early-exit trick is free
        for W in [1, 8, 32, 64, 128, 256]:
            n_iter = n_iter_for(32 * MB, 1, tile) if scaling == "weak" else n_iter_for(2 * GB, W, tile)
            cfg = BWConfig(op="read", mode="rr", W=W, n_iter=n_iter, plain_grid=True, **DEFAULT)
            out.add(run_cfg(cfg, bufs, flush, args.iters, sweep_tag=scaling + "_plain", scaling=scaling))
    out.save()


def sweep_B(bufs, flush, args):
    """Memory-level-parallelism controls at W=32/64, plus explicit CU packing via HWSEL."""
    out = CSVOut("B")
    total = GB
    for op, W, nw, u, wpc in itertools.product(["read", "write", "copy"], [32, 64], [1, 2, 4, 8, 16], [1, 2, 4, 8],
                                               [1, 2]):
        cfg = BWConfig(op=op, mode="rr", W=W, num_warps=nw, unroll=u, wpc=wpc)
        cfg = replace(cfg, n_iter=n_iter_for(total, W, cfg.tile_elems))
        out.add(run_cfg(cfg, bufs, flush, args.iters, sweep_tag="grid"))

    per_xcd, _ = discover_cus(bufs)
    cu32 = pick_cus(per_xcd, 4)
    cu64 = pick_cus(per_xcd, 8)
    for op, nw, u in itertools.product(["read", "write", "copy"], [1, 2, 4, 8], [1, 4]):
        cases = [
            ("32wg_32cu", BWConfig(mode="hwsel", W=32, num_warps=nw, hwsel_caps={k: 1 for k in cu32})),
            ("64wg_32cu", BWConfig(mode="hwsel", W=64, num_warps=nw, wpc=2, hwsel_caps={k: 2 for k in cu32})),
            ("64wg_64cu", BWConfig(mode="hwsel", W=64, num_warps=nw, hwsel_caps={k: 1 for k in cu64})),
            ("32wg_32cu_2xwaves", BWConfig(mode="hwsel", W=32, num_warps=2 * nw, hwsel_caps={k: 1 for k in cu32})),
            ("32wg_32cu_2xunroll", BWConfig(mode="hwsel", W=32, num_warps=nw, hwsel_caps={k: 1 for k in cu32})),
        ]
        for tag, cfg in cases:
            cfg = replace(cfg, op=op, unroll=2 * u if tag.endswith("2xunroll") else u)
            cfg = replace(cfg, n_iter=n_iter_for(total, cfg.W, cfg.tile_elems))
            out.add(run_cfg(cfg, bufs, flush, args.iters, sweep_tag=tag, base_warps=nw, base_unroll=u))
    out.save()


def sweep_C(bufs, flush, args):
    """Working-set sweep: L2 / MALL / HBM regimes, cold (flushed) vs warm (back-to-back)."""
    out = CSVOut("C")
    tile = BWConfig(**DEFAULT).tile_elems
    for op, W, cold in itertools.product(["read", "write", "copy"], [32, 64, 256], [True, False]):
        for size_mb in [16, 32, 64, 128, 192, 256, 384, 512, 1024, 2048, 4096]:
            n_iter = n_iter_for(size_mb * MB, W, tile)
            cfg = BWConfig(op=op, mode="rr", W=W, n_iter=n_iter, **DEFAULT)
            out.add(run_cfg(cfg, bufs, flush, args.iters, cold=cold, sweep_tag="cold" if cold else "warm",
                            size_mb=size_mb))
    out.save()


def sweep_D(bufs, flush, args):
    """XCD subsets: each single XCD, every pair, some quads, all 8 (32 CUs per XCD, 32 MB per worker)."""
    out = CSVOut("D")
    tile = BWConfig(**DEFAULT).tile_elems
    n_iter = n_iter_for(32 * MB, 1, tile)
    subsets = [(x, ) for x in range(8)] + list(itertools.combinations(range(8), 2)) + [
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 2, 4, 6), (1, 3, 5, 7), (0, 1, 4, 5), (2, 3, 6, 7), tuple(range(8))
    ]
    for op in ["read", "write", "copy"]:
        for sub in subsets:
            cfg = BWConfig(op=op, mode="subset", W=32 * len(sub), xcd_mask=sub, per_xcd=32, n_iter=n_iter, **DEFAULT)
            out.add(run_cfg(cfg, bufs, flush, args.iters, sweep_tag="subset", xcds="-".join(map(str, sub)),
                            n_sub=len(sub)))
    out.save()


def sweep_E(bufs, flush, args):
    """Access pattern: contiguous chunk per worker vs grid-stride, round-robin, weak scaling."""
    out = CSVOut("E")
    tile = BWConfig(**DEFAULT).tile_elems
    n_iter = n_iter_for(32 * MB, 1, tile)
    for op, pattern, W in itertools.product(["read", "write", "copy"], ["chunked", "grid"],
                                            [1, 2, 4, 8, 16, 32, 64, 128, 256]):
        cfg = BWConfig(op=op, mode="rr", pattern=pattern, W=W, n_iter=n_iter, **DEFAULT)
        out.add(run_cfg(cfg, bufs, flush, args.iters, sweep_tag=pattern))
    out.save()


def sweep_F(bufs, flush, args):
    """Split-k proxy: 32 output tiles, per-tile read R, split S, two-pass reduce vs atomics."""
    out = CSVOut("F")
    base = SplitKConfig(**DEFAULT)
    tile_bytes = base.tile_elems * 4
    max_read = 64 * MB * base.tiles
    src = bufs.src
    assert src.numel() * 4 >= max_read
    part = torch.empty(8 * base.tiles * base.out_repeat * base.tile_elems, dtype=torch.float32, device="cuda")
    res = torch.empty(base.tiles * base.out_repeat * base.tile_elems, dtype=torch.float32, device="cuda")
    for r_mb in [1, 4, 16, 64]:
        k_blocks = r_mb * MB // tile_bytes
        ref = None
        for S, variant in [(1, "direct"), (2, "twopass"), (4, "twopass"), (8, "twopass"), (2, "atomic"),
                           (4, "atomic"), (8, "atomic")]:
            cfg = replace(base, splits=S, k_blocks=k_blocks, variant=variant)
            launch = prepare_splitk(cfg, src, part, res)
            med, lo, hi = time_launch(launch, args.iters, flush)
            got = res[:cfg.out_elems].clone()
            if ref is None:
                ref = got
            ok = torch.allclose(got, ref, rtol=1e-4)
            partial_bytes = 0 if S == 1 else cfg.tiles * S * cfg.out_repeat * cfg.tile_elems * 4
            row = dict(sweep_tag=variant, op="splitk", mode="rr", W=cfg.tiles * S, splits=S, read_mb_per_tile=r_mb,
                       read_bytes=cfg.read_bytes, out_bytes=cfg.out_elems * 4, partial_bytes=partial_bytes,
                       time_ms=round(med, 4), time_min_ms=round(lo, 4), time_max_ms=round(hi, 4),
                       gbps=round(cfg.read_bytes / (med * 1e-3) / 1e9, 1), correct=int(ok))
            out.add(row)
    out.save()


# ----------------------------------------------------------------------------------------------
# Cache tiers: every point re-reads a fixed per-workgroup footprint inside one kernel (see run_passes)
# ----------------------------------------------------------------------------------------------
G_SIZES_KB = [16, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048, 4096]
H_SUBSETS = [(x, ) for x in range(8)] + [(0, 1), (0, 1, 2, 3), (4, 5, 6, 7), (0, 2, 4, 6), (0, 1, 4, 5), tuple(range(8))]
ALL8 = tuple(range(8))


def points_G():
    # 4 waves x UNROLL=4 -> 16 KB tiles, so the footprint can step in 16 KB increments
    return [(f"{op}_{kb}KB", footprint_cfg(op, kb, W=NUM_CUS)) for op in ["read", "write"] for kb in G_SIZES_KB]


def points_H():
    # 512 KB per CU: 16 MB per XCD (4x its L2), 128 MB total at most (inside the MALL). Default 8w x u4 as in sweep D.
    pts = [(f"xcds_{'-'.join(map(str, s))}", footprint_cfg("read", 512, mode="subset", W=32 * len(s), xcd_mask=s,
                                                             per_xcd=32, **DEFAULT)) for s in H_SUBSETS]
    # Fewer CUs per XCD: grow the per-CU footprint so every XCD still streams 16 MB and misses its L2.
    pts += [(f"all8_{n}cu", footprint_cfg("read", 16 * 1024 // n, mode="subset", W=8 * n, xcd_mask=ALL8, per_xcd=n,
                                          **DEFAULT)) for n in [1, 2, 4, 8, 16]]
    return pts


def points_I():
    # 64 KB per CU (2 MB per XCD): above the 32 KB L1, well inside the 4 MB L2. Tiles must not exceed 64 KB,
    # so the high-parallelism config is 16w x u4 (64 KB tile) rather than 16w x u8.
    pts = []
    for name, xs in [("x0", (0, )), ("all8", ALL8)]:
        for n in [1, 2, 4, 8, 16, 32]:
            for nw, u in [(4, 4), (8, 4), (16, 4)]:
                pts.append((f"{name}_{n}cu_{nw}w_u{u}", footprint_cfg("read", 64, num_warps=nw, unroll=u, mode="subset",
                                                                       W=len(xs) * n, xcd_mask=xs, per_xcd=n)))
    return pts


def points_J():
    return [(tier, footprint_cfg("read", kb, W=NUM_CUS, **DEFAULT))
            for tier, kb in [("L2_64KB", 64), ("MALL_512KB", 512), ("HBM_4096KB", 4096)]]


POINTS = dict(G=points_G, H=points_H, I=points_I, J=points_J)


def sweep_G(bufs, flush, args):
    """Tier sweep: steady-state bandwidth vs per-CU footprint (L1 / L2 / MALL / HBM), W=256 round-robin."""
    out = CSVOut("G")
    for tag, cfg in points_G():
        out.add(run_passes(cfg, bufs, flush, args.iters, sweep_tag=tag, footprint_total_mb=cfg.footprint_elems * 4 /
                           MB))
    out.save()


def sweep_H(bufs, flush, args):
    """MALL placement: steady-state MALL-hit bandwidth for XCD subsets and CUs per XCD."""
    out = CSVOut("H")
    for tag, cfg in points_H():
        out.add(run_passes(cfg, bufs, flush, args.iters, sweep_tag=tag, xcds="-".join(map(str, cfg.xcd_mask)),
                           n_sub=len(cfg.xcd_mask), cus_per_xcd=cfg.per_xcd))
    out.save()


def sweep_I(bufs, flush, args):
    """L2 scaling: steady-state L2-hit bandwidth vs CUs per XCD, one XCD vs all 8."""
    out = CSVOut("I")
    for tag, cfg in points_I():
        out.add(run_passes(cfg, bufs, flush, args.iters, sweep_tag=tag, n_sub=len(cfg.xcd_mask),
                           cus_per_xcd=cfg.per_xcd))
    out.save()


class L2Evictor:
    """Evicts every XCD's L2 but not the target data from the MALL: streams 64 MB of other data on all 256 CUs."""

    def __init__(self):
        self.bufs = Buffers(64 * MB // 4, dst_elems=1)
        cfg = BWConfig(op="read", W=NUM_CUS, n_iter=(64 * MB // 4) // (NUM_CUS * BWConfig(**DEFAULT).tile_elems),
                       record=False, **DEFAULT)
        self.launch = prepare(cfg, self.bufs)

    def __call__(self):
        self.launch()


def sweep_J(bufs, flush, args):
    """Kernel-boundary retention: one pass cold vs warm, against a second pass inside the same kernel."""
    out = CSVOut("J")
    l2_evict = L2Evictor()
    for tier, cfg in points_J():
        p1, p2 = with_passes(cfg, 1), with_passes(cfg, 2)
        cold = run_cfg(p1, bufs, flush, args.iters, cold=True, sweep_tag=tier, variant="cold_1pass")
        warm = run_cfg(p1, bufs, flush, args.iters, cold=False, sweep_tag=tier, variant="warm_1pass")
        # Same data still in the MALL, but each XCD's L2 overwritten by the evictor between launches.
        l2ev = run_cfg(p1, bufs, l2_evict, args.iters, cold=True, sweep_tag=tier, variant="l2_evicted_1pass")
        two = run_cfg(p2, bufs, flush, args.iters, cold=True, sweep_tag=tier, variant="cold_2pass")
        # The second pass of the 2-pass kernel is guaranteed to find whatever the first pass left in the caches.
        second_us = two["kernel_span_us"] - cold["kernel_span_us"]
        for r in (cold, warm, l2ev, two):
            r["second_pass_us"] = round(second_us, 2)
            r["second_pass_gbps"] = round(p1.bytes_moved / (second_us * 1e-6) / 1e9, 1)
            out.add(r)
    out.save()


def run_point(bufs, spec, passes, reps):
    """Run one sweep point `reps` times back-to-back with no flush (for rocprofv3 --pmc)."""
    sweep, tag = spec.split(":", 1)
    cfg = dict(POINTS[sweep]())[tag]
    cfg = with_passes(cfg, passes)
    launch = prepare(cfg, bufs)
    for _ in range(reps):
        launch.pre()
        launch()
    torch.cuda.synchronize()
    rec = analyse_records(cfg, bufs)
    print(f"point {spec} passes={passes} footprint={cfg.footprint_elems * 4 / MB:.1f} MB "
          f"bytes/launch={cfg.bytes_moved / MB:.1f} MB in-kernel={cfg.bytes_moved / rec['kernel_span_us'] / 1e3:.1f} GB/s")


SWEEPS = dict(check=sweep_check, T=sweep_T, A=sweep_A, B=sweep_B, C=sweep_C, D=sweep_D, E=sweep_E, F=sweep_F,
              G=sweep_G, H=sweep_H, I=sweep_I, J=sweep_J)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", nargs="+", default=["check"], choices=list(SWEEPS) + ["all"])
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--full", action="store_true", help="sweep A: every W in 1..256")
    p.add_argument("--results-dir", default=None, help="output directory for CSVs (default: results/)")
    p.add_argument("--point", default=None, help="run one point, e.g. G:read_256KB, back-to-back (for rocprofv3)")
    p.add_argument("--passes", type=int, default=16, help="--point: passes over the footprint per launch")
    p.add_argument("--reps", type=int, default=5, help="--point: launches")
    args = p.parse_args()
    sweeps = list(SWEEPS) if "all" in args.sweep else args.sweep
    if args.results_dir:
        global RESULTS
        RESULTS = os.path.abspath(args.results_dir)

    print(torch.cuda.get_device_properties(0))
    # 8 GB per buffer covers weak scaling at W=256 (32 MB x 256) and the 4 GB working-set point.
    bufs = Buffers(2 * GB)
    if args.point:
        run_point(bufs, args.point, args.passes, args.reps)
        return
    flush = Flusher()
    for s in sweeps:
        print(f"===== sweep {s}: {SWEEPS[s].__doc__.strip()}")
        log_clocks(s)
        t = time.time()
        SWEEPS[s](bufs, flush, args)
        log_clocks(s)
        print(f"===== sweep {s} done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
