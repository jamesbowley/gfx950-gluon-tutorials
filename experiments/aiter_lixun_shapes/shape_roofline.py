#!/usr/bin/env python3
"""The roofline of a shape on MI355X for a given tile and tile schedule, from the bytes each memory level
must move with an ideal hit rate. CPU only: it replays the schedule, nothing runs on the GPU.

    python shape_roofline.py [--measured results/details_adhoc1018.csv results/details_adhoc1018_rot2048.csv]

Per kernel and shape:
- Compute: the time of the busiest XCD at the bf16 dense peak, counting whole tiles (ragged tiles and
  K rounded up to the kernel's K granularity are computed in full).
- L2 -> CU (Q_tiled): every tile's A and B strips, (BM + BN) x K, plus its C store.
- L2 misses, which go over the fabric to the MALL or HBM (Q_L2miss): ideal hit rate, i.e. the
  concurrently resident tiles of an XCD (one wave of 32 CUs, one workgroup each) run in K lockstep and share
  every block they have in common, so each distinct A row-block and B column-block in the window is
  fetched once per XCD per wave; no reuse across waves (a block's K strip is 1.5-2 MB, so a window's
  strips are far past the 4 MB L2). In-bounds bytes only, plus C written once.
- HBM (Q_comp): A, B and C once. A + B fits in the 256 MB MALL for both shapes, so with an ideal hit
  rate every other fetch is a MALL hit. Back to back on the same buffers (warm) A and B can stay in the
  MALL between launches, leaving only C's write-back.
The roof is the useful FLOPs over the slowest level, every level overlapping perfectly.
"""

import argparse
import csv
import math
from collections import defaultdict

PEAK = 2516.6e12  # bf16 dense MFMA, 256 CUs at 2.4 GHz (spec)
CUS, XCDS = 256, 8
CU_PER_XCD = CUS // XCDS
# experiments/hbm_bandwidth_xcd (this machine): chip-wide reads.
BW_L2_CU = 38.8e12  # L2 hits, 2 KB/clk per XCD
BW_FABRIC = 7.5e12  # L2 misses served by the MALL (7.0-7.9); HBM misses are below this anyway
BW_FABRIC_XCD = 1.35e12
BW_HBM = 6.3e12  # 6.2-6.4
E = 2  # bf16


def ours_schedule(M, N, BM, BN, gm=4):
    """v26 / v27 persistent loop and v9 / v9_ragged / v9_tile / new AITER v9 dispatch: XCD x owns tile ids
    [x * per, (x + 1) * per) (xcd_remap_tiles / get_pids), runs them 32 at a time in local order, and the
    GROUP_SIZE_M swizzle turns ids into (m, n). Returns {xcd: [window, ...]}, a window being [(m, n), ...]."""
    tm, tn = math.ceil(M / BM), math.ceil(N / BN)
    total = tm * tn
    per = math.ceil(total / XCDS)
    tall = total % XCDS or XCDS

    def swizzle(t):
        in_group = gm * tn
        g = t // in_group
        first = g * gm
        size = min(tm - first, gm)
        return first + (t % in_group) % size, (t % in_group) // size

    out = {}
    for x in range(XCDS):
        base = x * per if x < tall else tall * per + (x - tall) * (per - 1)
        count = per if x < tall else per - 1
        ids = [base + i for i in range(count)]
        out[x] = [[swizzle(t) for t in ids[w:w + CU_PER_XCD]] for w in range(0, count, CU_PER_XCD)]
    return out


def hbl_schedule(M, N, BM, BN):
    """hipBLASLt as assumed: one workgroup per tile, workgroup w on XCD w % 8 (SKXCCM0: no XCC remap), the
    tile index along hipBLASLt's first dimension (torch's N) fastest (WGM = 1, not in the solution name),
    and 256 workgroups resident at a time."""
    tm, tn = math.ceil(M / BM), math.ceil(N / BN)
    total = tm * tn
    out = defaultdict(list)
    for w0 in range(0, total, CUS):
        per = defaultdict(list)
        for w in range(w0, min(w0 + CUS, total)):
            per[w % XCDS].append((w // tn, w % tn))
        for x, tiles in per.items():
            out[x].append(tiles)
    return dict(out)


def model(M, N, K, BM, BN, sched, k_gran):
    def bm_eff(m):
        return min(BM, M - m * BM)

    def bn_eff(n):
        return min(BN, N - n * BN)

    flops = 2.0 * M * N * K
    k_pad = math.ceil(K / k_gran) * k_gran
    t_tile = BM * BN * k_pad * 2 / (PEAK / CUS)
    waves = max(len(ws) for ws in sched.values())
    t_comp = waves * t_tile
    q_tiled = q_miss = 0
    miss_x = defaultdict(float)
    for x, ws in sched.items():
        for w in ws:
            q_tiled += sum((BM + BN) * K * E + bm_eff(m) * bn_eff(n) * E for m, n in w)
            ms, ns = {m for m, _ in w}, {n for _, n in w}
            b = (sum(bm_eff(m) for m in ms) + sum(bn_eff(n) for n in ns)) * K * E
            b += sum(bm_eff(m) * bn_eff(n) * E for m, n in w)
            q_miss += b
            miss_x[x] += b
    q_comp = (M * K + N * K + M * N) * E
    q_c = M * N * E
    t = dict(compute=t_comp, l2_cu=q_tiled / BW_L2_CU,
             fabric=max(q_miss / BW_FABRIC, max(miss_x.values()) / BW_FABRIC_XCD),
             hbm_cold=q_comp / BW_HBM)
    tiles = sum(len(w) for ws in sched.values() for w in ws)
    return dict(flops=flops, tiles=tiles, waves=waves, k_pad=k_pad, t=t, q_tiled=q_tiled, q_miss=q_miss,
                q_miss_xcd_max=max(miss_x.values()), q_comp=q_comp, q_c=q_c,
                hit=1 - (q_miss - q_c) / (q_tiled - tiles * 0 - q_c) if q_tiled > q_c else float("nan"))


def measured(paths):
    out = {}
    for state, path in zip(("warm", "cold"), paths):
        for r in csv.DictReader(open(path)):
            k = (int(r["M"]), int(r["N"]), int(r["K"]))
            f = lambda c: float(r[c]) if r.get(c) not in (None, "", "nan") else float("nan")
            out[(k, state)] = dict(hbl=f("hbl_tflops"), pick=f("aiter_pick_tflops"), v27=f("v20_tflops"),
                                   v9new=f("v9new_tflops"), v26=f("v26_tflops"), v27dp=f("v20_dp_tflops"))
    return out


CASES = [
    # shape, label, tile (torch BM, BN), schedule, K granularity, measured columns
    ((8192, 5120, 2880), "ours: v26, v27, new AITER v9 (256x256, GM 4)", (256, 256), "ours", 128,
     ("v26", "v27", "v9new")),
    ((8192, 5120, 2880), "hipBLASLt MT256x224x64 (224x256 in torch's M x N)", (224, 256), "hbl", 64, ("hbl",)),
    ((4096, 2880, 4096), "ours: v26, v27 (-> v9_tile), new AITER v9 (256x256, GM 4)", (256, 256), "ours", 128,
     ("v26", "v27", "v9new")),
    ((4096, 2880, 4096), "hipBLASLt MT192x256x64 (256x192 in torch's M x N)", (256, 192), "hbl", 64, ("hbl",)),
    ((4096, 2880, 4096), "what if: our schedule on hipBLASLt's 256x192 tile", (256, 192), "ours", 128, ()),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measured", nargs=2, default=("results/details_adhoc1018.csv",
                                                   "results/details_adhoc1018_rot2048.csv"))
    a = p.parse_args()
    meas = measured(a.measured)
    for (M, N, K), label, (BM, BN), kind, kg, cols in CASES:
        sched = ours_schedule(M, N, BM, BN) if kind == "ours" else hbl_schedule(M, N, BM, BN)
        r = model(M, N, K, BM, BN, sched, kg)
        t = r["t"]
        bind = max(t, key=t.get)
        roof = r["flops"] / t[bind]
        warm_t = dict(t, hbm_cold=r["q_c"] / BW_HBM)
        roof_warm = r["flops"] / max(warm_t.values())
        print(f"\n## {M}x{N}x{K}: {label}")
        print(f"tiles {r['tiles']}, waves per XCD {r['waves']}, K computed {r['k_pad']}; FLOPs {r['flops']:.4g}")
        print(f"bytes: L2->CU {r['q_tiled'] / 1e6:.0f} MB, L2 misses {r['q_miss'] / 1e6:.0f} MB "
              f"(busiest XCD {r['q_miss_xcd_max'] / 1e6:.0f} MB; ideal L2 hit rate on operands "
              f"{100 * r['hit']:.1f}%), HBM {r['q_comp'] / 1e6:.0f} MB cold / {r['q_c'] / 1e6:.0f} MB warm")
        print("time at each level (us): " + ", ".join(f"{k} {v * 1e6:.1f}" for k, v in t.items()))
        print(f"roof: {roof / 1e12:.0f} TFLOPS ({100 * roof / PEAK:.1f}% of peak), bound by {bind}; "
              f"warm roof {roof_warm / 1e12:.0f} TFLOPS")
        for st in ("warm", "cold"):
            m = meas.get(((M, N, K), st))
            if m and cols:
                rf = roof if st == "cold" else roof_warm
                print(f"  measured {st}: " + ", ".join(
                    f"{c} {m[c]:.0f} TFLOPS = {100 * m[c] * 1e12 / rf:.0f}% of roof" for c in cols if m[c] == m[c]))


if __name__ == "__main__":
    main()
