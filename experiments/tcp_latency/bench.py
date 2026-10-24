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
"""Sweeps for the load-latency and loads-in-flight experiment.

    HIP_VISIBLE_DEVICES=2 python bench.py --sweep lat loaded inflight

    lat       one wave, one 1 KB load in flight, per cache tier (L1, L2, MALL, HBM)
    loaded    the same probe while W other workgroups stream from HBM or L2, or run MFMAs
    clock     the probe at the lower clocks MFMA-only co-runners cause (ns-fixed or cycle-fixed latency?)
    inflight  bandwidth per CU against KB in flight per CU, on one CU and on all 256

Each sweep writes results/<sweep>.csv; lat, loaded and clock also write results/<sweep>_samples.npz (per
load: cycles, s_memtime, s_memrealtime; keyed by run tag).
"""

import argparse
import csv
import itertools
import os
import statistics
import subprocess
import time
from dataclasses import replace

import numpy as np
import torch
from kernel import (
    LDS_FOR_WPC,
    NUM_CUS,
    NUM_XCDS,
    PROBE_REC,
    PROBE_TILE,
    Buffers,
    BWConfig,
    att_stream_kernel,
    prepare,
    probe_kernel,
    read_records,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
KB = 1 << 10
MB = 1 << 20
GB = 1 << 30
TILE_BYTES = PROBE_TILE * 4

# Streamers in the loaded sweep: 8 waves x 4 x 16 B loads per thread, double-buffered: up to 64 KB in flight per CU.
STREAM = dict(num_warps=8, unroll=4)


# ----------------------------------------------------------------------------------------------
# Timing helpers (from experiments/hbm_bandwidth_xcd/bench.py)
# ----------------------------------------------------------------------------------------------
class Flusher:
    """Evicts L2 and MALL between timed runs by streaming 1 GB of reads (clean lines, no write-back)."""

    def __init__(self):
        self.bufs = Buffers(GB // 4, dst_elems=1)
        cfg = BWConfig(op="read", W=NUM_CUS, n_iter=(GB // 4) // (NUM_CUS * BWConfig(**STREAM).tile_elems),
                       record=False, **STREAM)
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
    """Placement, in-kernel timing and per-CU concurrency from the last launch's per-workgroup records."""
    r = read_records(cfg, bufs)
    act = r["active"]
    n_act = int(act.sum())
    if n_act == 0:
        return dict(n_active=0)
    keys = r["cu_key"][act]
    t0 = r["t0"][act].double()
    t1 = r["t1"][act].double()
    span_us = (t1.max() - t0.min()).item() / 100.0
    dt = (t1 - t0).clamp(min=1)
    clk = ((r["c1"][act] - r["c0"][act]).double() / dt * 100.0)
    uk, cu_counts = torch.unique(keys, return_counts=True)
    # Concurrency per CU: busy time of its workgroups over the CU's span. WPC if they all overlapped.
    conc = []
    for k in uk.tolist():
        m = keys == k
        conc.append(dt[m].sum().item() / max(1.0, (t1[m].max() - t0[m].min()).item()))
    return dict(
        n_active=n_act,
        n_cus=len(cu_counts),
        max_wg_per_cu=int(cu_counts.max()),
        conc_per_cu=round(statistics.mean(conc), 2),
        kernel_span_us=round(span_us, 2),
        wg_time_mean_us=round(dt.mean().item() / 100.0, 2),
        sclk_mhz=round(clk.median().item(), 0),
    )


def run_cfg(cfg, bufs, flush, iters, cold=True, **extra):
    launch = prepare(cfg, bufs)
    med, lo, hi = time_launch(launch, iters, flush, cold=cold)
    row = dict(extra)
    row.update(W=cfg.W, num_warps=cfg.num_warps, unroll=cfg.unroll, wpc=cfg.wpc, n_iter=cfg.n_iter,
               passes=cfg.passes, footprint_bytes=cfg.footprint_elems * 4, bytes=cfg.bytes_moved, cold=int(cold),
               time_ms=round(med, 4))
    rec = analyse_records(cfg, bufs)
    if rec.get("kernel_span_us"):
        rec["gbps_in_kernel"] = round(cfg.bytes_moved / (rec["kernel_span_us"] * 1e-6) / 1e9, 2)
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


def run_passes(cfg, bufs, flush, iters, **extra):
    """Steady-state bandwidth over cfg's footprint from the difference between P and 4P passes (in-kernel spans)."""
    P = pick_passes(cfg, bufs)
    lo_cfg, hi_cfg = with_passes(cfg, P), with_passes(cfg, 4 * P)
    lo = run_cfg(lo_cfg, bufs, flush, iters)
    hi = run_cfg(hi_cfg, bufs, flush, iters)
    row = dict(extra)
    row.update({k: hi[k] for k in hi if k not in ("time_ms", "bytes", "gbps_in_kernel")})
    d_bytes = hi_cfg.bytes_moved - lo_cfg.bytes_moved
    d_us = hi["kernel_span_us"] - lo["kernel_span_us"]
    row.update(P=P, bytes=d_bytes, gbps=round(d_bytes / (d_us * 1e-6) / 1e9, 2))
    return row


class CSVOut:

    def __init__(self, name):
        os.makedirs(RESULTS, exist_ok=True)
        self.path = os.path.join(RESULTS, f"{name}.csv")
        self.rows = []

    def add(self, row, keys):
        self.rows.append(row)
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
    with open(os.path.join(RESULTS, f"{name}_clocks.txt"), "a") as f:
        f.write(f"=== {time.ctime()} (physical GPU {gpu})\n{txt}\n")


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
    return per_xcd


# ----------------------------------------------------------------------------------------------
# Latency probe
# ----------------------------------------------------------------------------------------------
class Probe:
    """Buffers for probe_kernel. The probe region and the stream region never overlap."""

    def __init__(self, probe_mb=256, stream_mb=1024, max_steps=1 << 17):
        self.src = torch.ones(probe_mb * MB // 4, dtype=torch.float32, device="cuda")
        self.ssrc = torch.ones(stream_mb * MB // 4, dtype=torch.float32, device="cuda")
        self.samples = torch.zeros(3 * max_steps, dtype=torch.int64, device="cuda")
        self.prec = torch.zeros(8, dtype=torch.int64, device="cuda")
        self.pout = torch.zeros(NUM_CUS + 1, dtype=torch.float32, device="cuda")
        self.srec = torch.zeros((NUM_CUS + 1) * 3, dtype=torch.int64, device="cuda")
        self.flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.probe_tiles = self.src.numel() // PROBE_TILE

    def launch(self, region, passes, start=0, W=0, stream_tier="hbm", num_warps=1, n_sleep=0):
        n_steps = region * passes
        assert 3 * n_steps <= self.samples.numel() and start + region <= self.probe_tiles
        stile = STREAM["num_warps"] * STREAM["unroll"] * 64 * 4  # fp32 elements per streamer iteration
        if W == 0:
            s_wrap = 1
        elif stream_tier == "hbm":
            s_wrap = self.ssrc.numel() // (W * stile)  # the whole 1 GB buffer: misses the 256 MB MALL
        else:
            s_wrap = 2  # 64 KB per streamer: 2 MB per XCD, inside its 4 MB L2
        self.flag.zero_()
        self.samples.zero_()
        a, b = LDS_FOR_WPC[1]
        probe_kernel[(1 + W, )](self.src, self.samples, self.prec, self.pout, n_steps, region, start, n_sleep,
                                self.ssrc, self.srec, self.flag, W, s_wrap, 1 << 30, UNROLL=STREAM["unroll"],
                                LDS_A=a, LDS_B=b, STREAM_MFMA=stream_tier == "mfma", num_warps=num_warps)
        torch.cuda.synchronize()
        s = self.samples[:3 * n_steps].view(n_steps, 3).cpu().numpy()  # cycles, s_memtime, s_memrealtime per load
        p = self.prec[:PROBE_REC].cpu().tolist()
        xcc, hw, rt0, rt1, ck0, ck1 = p
        out = dict(probe_xcc=xcc & 15, probe_cu_key=(xcc & 15) * 256 + ((hw >> 13) & 7) * 32 + ((hw >> 12) & 1) * 16 +
                   ((hw >> 8) & 15), probe_us=round((rt1 - rt0) / 100.0, 1),
                   probe_ghz=round((ck1 - ck0) / max(1, rt1 - rt0) / 10.0, 3))
        if W > 0:
            sr = self.srec[3:3 * (W + 1)].view(W, 3).cpu()
            it, st0, st1 = sr[:, 0].double(), sr[:, 1].double(), sr[:, 2].double()
            span = (st1.max() - st0.min()).item() / 100.0
            gbps = 0.0 if stream_tier == "mfma" else (it * stile * 4).sum().item() / (span * 1e-6) / 1e9
            out.update(stream_us=round(span, 1), stream_gbps=round(gbps, 1),
                       stream_covers_probe=int(st0.max().item() <= rt0 and st1.min().item() >= rt1))
        return s, out


def local_ghz(s, half=32):
    """Shader clock around each load, from s_memtime / s_memrealtime over +-half loads (the clock moves under load)."""
    n = len(s)
    i = np.arange(n)
    lo, hi = np.clip(i - half, 0, n - 1), np.clip(i + half, 0, n - 1)
    d_ck = (s[hi, 1] - s[lo, 1]).astype(np.float64)
    d_rt = (s[hi, 2] - s[lo, 2]).astype(np.float64)
    return d_ck / np.maximum(d_rt, 1) / 10.0


def summarize(s):
    """Percentiles of one tier's loads in cycles and in ns at the local clock. s: (n, 3) rows from Probe.launch."""
    c = s[:, 0].astype(np.float64)
    ns = c / local_ghz(s)
    q = np.percentile(c, [10, 50, 90])
    qn = np.percentile(ns, [10, 50, 90])
    return dict(n=len(c), local_ghz=round(float(np.median(local_ghz(s))), 3), p10_cyc=round(q[0], 0),
                p50_cyc=round(q[1], 0), p90_cyc=round(q[2], 0), mean_cyc=round(c.mean(), 1), p10_ns=round(qn[0], 1),
                p50_ns=round(qn[1], 1), p90_ns=round(qn[2], 1), mean_ns=round(ns.mean(), 1))


# Tier runs: (tag, region tiles, passes, {pass index: tier}). Pass 0 after a flush is a first touch: HBM. A 1 MB region
# misses the 32 KB L1 and fits the 4 MB L2; 16 MB misses L2 and fits the 256 MB MALL.
TIER_RUNS = [
    ("l1_16KB", 16, 64, {p: "L1" for p in range(1, 64)}),
    ("l2_1MB", 1024, 4, {0: "HBM", 1: "L2", 2: "L2", 3: "L2"}),
    ("mall_16MB", 16384, 2, {0: "HBM", 1: "MALL"}),
]
LAT_KEYS = ["tag", "tier", "W", "stream", "num_warps", "local_ghz", "p10_cyc", "p50_cyc", "p90_cyc", "p50_ns",
            "stream_gbps", "stream_covers_probe"]


def run_probe_rows(probe, flush, out, store, tag_prefix, reps, **kw):
    for tag, region, passes, tiers in TIER_RUNS:
        for rep in range(reps):
            flush()
            torch.cuda.synchronize()
            s, info = probe.launch(region, passes, **kw)
            key = f"{tag_prefix}{tag}_r{rep}"
            store[key] = s
            by_tier = {}
            for p, tier in tiers.items():
                by_tier.setdefault(tier, []).append(s[p * region:(p + 1) * region])
            for tier, parts in by_tier.items():
                row = dict(tag=key, run=tag, tier=tier, rep=rep, W=kw.get("W", 0), stream=kw.get("stream_tier", ""),
                           num_warps=kw.get("num_warps", 1))
                row.update(info)
                row.update(summarize(np.concatenate(parts)))
                out.add(row, LAT_KEYS)


def sweep_lat(args):
    """Unloaded: one wave, one load in flight, every tier, on one wave (num_warps=1) and with the masked 8-wave probe."""
    out = CSVOut("lat")
    store = {}
    probe = Probe()
    flush = Flusher()
    for nw in (1, 8):
        run_probe_rows(probe, flush, out, store, f"nw{nw}_", args.reps, num_warps=nw)
    out.save()
    np.savez_compressed(os.path.join(RESULTS, "lat_samples.npz"), **store)


def sweep_loaded(args):
    """Probe on workgroup 0 while workgroups 1..W (one per CU) stream from HBM (1 GB) or L2 (64 KB each), or run MFMAs."""
    out = CSVOut("loaded")
    store = {}
    probe = Probe()
    flush = Flusher()
    for stream_tier, W in itertools.product(["hbm", "l2", "mfma"], [0, 8, 32, 64, 128, 255]):
        # 10 x s_sleep 127 (~35 us) lets the streamers ramp before the probe's first-touch (HBM) pass. The MFMA
        # co-runner needs longer for the clock to settle under the power cap: 300 (~1 ms at 2.4 GHz).
        run_probe_rows(probe, flush, out, store, f"{stream_tier}_W{W}_", args.reps, W=W, stream_tier=stream_tier,
                       num_warps=STREAM["num_warps"], n_sleep=300 if stream_tier == "mfma" else 10)
    out.save()
    np.savez_compressed(os.path.join(RESULTS, "loaded_samples.npz"), **store)


def sweep_clock(args):
    """Is latency fixed in ns or in cycles? MFMA-only co-runners on W CUs pull the clock down under the power cap
    without touching memory; ~0.1 s of s_sleep lets it settle before the probe starts."""
    out = CSVOut("clock")
    store = {}
    probe = Probe()
    flush = Flusher()
    for W in [0, 64, 128, 192, 255]:
        run_probe_rows(probe, flush, out, store, f"mfma_W{W}_", args.reps, W=W, stream_tier="mfma",
                       num_warps=STREAM["num_warps"], n_sleep=30000)
    out.save()
    np.savez_compressed(os.path.join(RESULTS, "clock_samples.npz"), **store)


# ----------------------------------------------------------------------------------------------
# KB in flight per CU against bandwidth per CU
# ----------------------------------------------------------------------------------------------
def inflight_points(max_kb=128):
    """(num_warps, unroll, wpc) per nominal KB in flight per CU = 2 x waves x unroll x 1 KB (register double-buffer).

    No num_warps=1: HWSEL placement does not hold at one wave per workgroup (every candidate workgroup goes active).
    """
    pts = []
    for nw, u, wpc in itertools.product([2, 4, 8, 16], [1, 2, 4], sorted(LDS_FOR_WPC)):
        kb = 2 * nw * u * wpc
        if nw * wpc <= 32 and kb <= max_kb:
            pts.append((nw, u, wpc, kb))
    return sorted(pts, key=lambda p: (p[3], p[2], -p[0]))


# Per-CU footprints: (tier, KB per CU on one CU, KB per CU on all 256 CUs, steady-state passes?). HBM is one cold pass.
INFLIGHT_TIERS = [("L2", 1024, 64, True), ("MALL", 16 * 1024, 512, True), ("HBM", 32 * 1024, 4 * 1024, False)]
INFLIGHT_KEYS = ["scope", "tier", "kb_inflight", "num_warps", "unroll", "wpc", "conc_per_cu", "sclk_mhz", "gbps_per_cu",
                 "bpc_per_cu"]


def sweep_inflight(args):
    """Bandwidth per CU against nominal KB in flight per CU, on one CU (no contention) and on all 256 CUs."""
    out = CSVOut("inflight")
    bufs = Buffers(2 * GB)
    flush = Flusher()
    per_xcd = discover_cus(bufs)
    one = per_xcd[0][0]
    for scope, (tier, kb1, kb256, steady), (nw, u, wpc, kb) in itertools.product(["1cu", "256cu"], INFLIGHT_TIERS,
                                                                               inflight_points()):
        cus = [one] if scope == "1cu" else [k for v in per_xcd.values() for k in v]
        per_cu_kb = kb1 if scope == "1cu" else kb256
        cfg = BWConfig(op="read", mode="hwsel", W=len(cus) * wpc, num_warps=nw, unroll=u, wpc=wpc,
                       hwsel_caps={k: wpc for k in cus})
        wrap = max(1, per_cu_kb * KB // wpc // (cfg.tile_elems * 4))
        cfg = replace(cfg, n_iter=wrap, passes=1)
        extra = dict(scope=scope, tier=tier, kb_inflight=kb, per_cu_kb=per_cu_kb)
        if steady:
            row = run_passes(cfg, bufs, flush, args.iters, **extra)
        else:
            row = run_cfg(cfg, bufs, flush, args.iters, cold=True, **extra)
            row["gbps"] = row.get("gbps_in_kernel")
        n_cus = row.get("n_cus") or len(cus)
        row["gbps_per_cu"] = round(row["gbps"] / n_cus, 2)
        row["bpc_per_cu"] = round(row["gbps_per_cu"] / (row["sclk_mhz"] / 1000.0), 2)
        out.add(row, INFLIGHT_KEYS)
    out.save()


def att_point(spec, reps):
    """Run under rocprofv3 --att: att_stream_kernel, one workgroup of num_warps x unroll per CU, on the 32 CUs of
    XCD 0 ("xcd0") or all 256 ("all"), each reading a per-CU footprint of the given tier. HBM is cold: a torch read
    of 1 GB evicts L2 and MALL without a second Triton kernel in the process, so ATT's kernel_iteration_range
    counts only measured launches."""
    nw, u, tier, scope = spec.split(",")
    nw, u = int(nw), int(u)
    per_cu_kb = dict(L2=1024, MALL=16 * 1024, HBM=32 * 1024)[tier]
    tile = 4 * 64 * nw * u
    wrap = max(1, per_cu_kb * KB // (tile * 4))
    n_iter = wrap * (1 if tier == "HBM" else 4)
    src = torch.ones(NUM_CUS * wrap * tile, dtype=torch.float32, device="cuda")
    out = torch.empty(NUM_CUS, dtype=torch.float32, device="cuda")
    fb = torch.ones(GB // 4, dtype=torch.float32, device="cuda")
    a, b = LDS_FOR_WPC[1]
    ts = []
    for _ in range(reps):
        if tier == "HBM":
            fb.sum()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        att_stream_kernel[(NUM_CUS, )](src, out, n_iter, wrap, UNROLL=u, LDS_A=a, LDS_B=b, XCD0_ONLY=scope == "xcd0",
                                       num_warps=nw)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ms = statistics.median(ts)
    print(f"att point nw={nw} u={u} tier={tier} scope={scope}: {ms:.3f} ms, "
          f"{n_iter * tile * 4 / (ms * 1e-3) / 1e9:.1f} GB/s per CU (event timing)")


SWEEPS = dict(lat=sweep_lat, loaded=sweep_loaded, clock=sweep_clock, inflight=sweep_inflight)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", nargs="+", default=["lat"], choices=list(SWEEPS))
    p.add_argument("--iters", type=int, default=5, help="inflight: timed launches per point (median)")
    p.add_argument("--reps", type=int, default=3, help="lat / loaded: launches per probe run")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--att-point", default=None,
                   help="'num_warps,unroll,tier,xcd0|all': run one att_stream_kernel config for rocprofv3 --att")
    args = p.parse_args()
    if args.results_dir:
        global RESULTS
        RESULTS = os.path.abspath(args.results_dir)
    os.makedirs(RESULTS, exist_ok=True)
    print(torch.cuda.get_device_properties(0))
    if args.att_point:
        att_point(args.att_point, args.reps * 6)
        return
    for s in args.sweep:
        print(f"===== sweep {s}: {SWEEPS[s].__doc__.strip()}")
        log_clocks(s)
        t = time.time()
        SWEEPS[s](args)
        log_clocks(s)
        print(f"===== sweep {s} done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
