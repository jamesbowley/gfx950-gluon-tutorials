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
"""Stream-K cost sweeps on a one-wave 4096x4096xK GEMM (256 tiles, one per program).

    HIP_VISIBLE_DEVICES=2 python bench.py --sweep check kstep store read switch

Needs the v10+ stack (see kernels/gemm/intra_wave/a16w16/regress.py) without the amdgcnas
peephole, which faults on these builds' spill code:
    export PYTHONPATH=<repo>/scripts/triton_deepbind_shim:<triton_gfx950-tutorial-v2.2>/python
    export LLVM_PASS_PLUGIN_PATH=<repo>/plugins/llir_scheduler/libLlirSched.so
    unset TRITON_AMDGCNAS_PLUGIN

Every point is the median per-launch time of back-to-back launches queued behind a GPU
sleep, so neither host launch overhead nor do_bench's 256 MB cache-flush memset is in the
number (both inflate a ~170 us kernel by 5-10 us). Costs are slopes over a count (partials
stored, partials read, tile switches) and are converted to k-steps with the kstep slope.
"""

import argparse
import os
import statistics
import sys
import tempfile

if os.environ.get("LLVM_PASS_PLUGIN_PATH"):
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)

import numpy as np
import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "kernels", "gemm", "utils"))
sys.path.insert(0, os.path.join(REPO, "experiments", "aiter_bias_spills"))

import kernel as K_
from check_spills import parse_amdgcn

M = N = 4096
K_MAIN = 8192
KSTEP_KS = [1024, 2048, 4096, 6144, 8192, 10240, 12288]
N_LAUNCH = 100
STORERS = {"256": (1, 0), "240": (16, 1), "16": (16, 0)}


def inputs(Kdim, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.rand((M, Kdim), device="cuda", dtype=torch.bfloat16, generator=g) - 0.5
    b = (torch.rand((N, Kdim), device="cuda", dtype=torch.bfloat16, generator=g) - 0.5).T
    c = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    return a, b, c


def time_us(launch, n=N_LAUNCH):
    for _ in range(5):
        launch()
    torch.cuda.synchronize()
    s = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    # The host needs ~150 us to enqueue one launch; the sleep lets it run ahead.
    torch.cuda._sleep(int(2.4e9 * n * 400e-6))
    for i in range(n):
        s[i].record()
        launch()
        e[i].record()
    torch.cuda.synchronize()
    return statistics.median(x.elapsed_time(y) for x, y in zip(s, e)) * 1e3


def fit(xs, ys):
    slope, icpt = np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)
    resid = np.asarray(ys) - (slope * np.asarray(xs) + icpt)
    return slope, icpt, float(np.max(np.abs(resid)))


REG_SEEN = {}


def regs(launch, key):
    """VGPR / AGPR / spill counts of the kernel `launch` runs. Roles are runtime arguments, so
    there is one specialization per (K, SEGMENTS)."""
    if key not in REG_SEEN:
        with tempfile.NamedTemporaryFile("w", suffix=".amdgcn", delete=False) as f:
            f.write(launch().asm["amdgcn"])
        REG_SEEN[key] = parse_amdgcn(f.name)
        os.unlink(f.name)
    return REG_SEEN[key]


def fmt_regs(r):
    if not r:
        return "regs ?"
    s = r.get("spill_ops", {})
    return (f"VGPR {r.get('vgpr_count', '?')} AGPR {r.get('agpr_count', '?')} "
            f"spill {r.get('vgpr_spill', '?')}/{r.get('sgpr_spill', '?')} "
            f"scratch {s.get('kloop', '?')}/{s.get('tile_loop', '?')}/{s.get('outside', '?')}")


# ---------------------------------------------------------------------------------------------
# Correctness of the knobs: segment offsets, the partial round trip and the fixup sum.
# ---------------------------------------------------------------------------------------------


def ref_tiles(a, b):
    ref = torch.matmul(a.float(), b.float())
    tiles = K_.tile_of_spid(M, N)
    out = {}
    for spid, (tm, tn) in tiles.items():
        out[spid] = ref[tm * 256:(tm + 1) * 256, tn * 256:(tn + 1) * 256]
    return ref, out, tiles


def quadrants(t):
    return [t[:128, :128], t[128:, :128], t[:128, 128:], t[128:, 128:]]


def lane_layout_index():
    """Row-major (row, col) -> element offset of kernel.quad_offsets' lane-contiguous layout."""
    r = torch.arange(128)[:, None]
    c = torch.arange(128)[None, :]
    f = ((r & 15) << 2) | (((r >> 4) & 1) << 9) | (((r >> 5) & 1) << 12) | (((r >> 6) & 1) << 13)
    g = (c & 3) | (((c >> 2) & 3) << 6) | (((c >> 4) & 1) << 8) | (((c >> 5) & 1) << 10) | (((c >> 6) & 1) << 11)
    return (f + g).cuda()


def check():
    a, b, c = inputs(K_MAIN)
    P, locks = K_.make_workspace()
    ok = True

    def report(name, good):
        nonlocal ok
        ok &= good
        print(f"  {name}: {'ok' if good else 'FAIL'}", flush=True)

    ref, rt, tiles = ref_tiles(a, b)
    K_.launcher(a, b, c, P, locks)()
    report("baseline C == A @ B", torch.allclose(c.float(), ref, atol=1e-1, rtol=1e-2))

    half = K_MAIN // 2
    ref2 = torch.matmul(a[:, half:].float(), b[half:, :].float())
    for mode, sw, rel in [("none", (0, 0), 1), ("c", (0, 1), 1), ("partial", (1, 0), 1)]:
        c.zero_()
        K_.launcher(a, b, c, P, locks, mode="switch", segments=2, sw=sw, release=rel)()
        report(f"2 segments, switch={mode}: C == second half of K",
               torch.allclose(c.float(), ref2, atol=1e-1, rtol=1e-2))

    idx = lane_layout_index()
    assert torch.equal(idx.flatten().sort().values, torch.arange(128 * 128, device="cuda"))
    for layout in (0, 1):
        P.zero_()
        K_.launcher(a, b, c, P, locks, mode="store", fin=(1, 0, 1), p_layout=layout)()
        Pv = P.view(K_.NUM_PROGRAMS, K_.MAX_REPS, 4, 128 * 128)
        good = True
        for s in range(K_.NUM_PROGRAMS):
            for q in range(4):
                got = Pv[s, 0, q].view(128, 128) if layout == 0 else Pv[s, 0, q][idx]
                good &= torch.allclose(got, quadrants(rt[s])[q], atol=1e-2, rtol=1e-3)
        report(f"partial store, P layout {layout}: P[spid] == fp32 tile", good)

    for s, sc, rc, lay, depth in ((2, ".wt", ".cv", 0, 1), (4, ".wt", ".cv", 0, 1), (16, ".wt", ".cv", 0, 1),
                                  (16, "", "", 0, 1), (16, ".wt", ".cv", 1, 1), (8, "", "", 1, 1)):
        locks.zero_()
        c.zero_()
        P.zero_()
        K_.launcher(a, b, c, P, locks, mode="fixup", sel_mod=s, fin=(0, s - 1, 1), fin_u=(1, 0, 0),
                    reset_flags=1, store_cache=sc, read_cache=rc, p_layout=lay, read_depth=depth)()
        good = True
        for o in range(0, K_.NUM_PROGRAMS, s):
            want = sum(rt[o + j] for j in range(s))
            tm, tn = tiles[o]
            got = c[tm * 256:(tm + 1) * 256, tn * 256:(tn + 1) * 256].float()
            good &= torch.allclose(got, want, atol=2e-1, rtol=2e-2)
        report(f"fixup s={s} store '{sc}' read '{rc}' layout {lay} depth {depth}: owner C == sum of group "
               "tiles; flags re-armed",
               good and int(locks.sum()) == 0)
    for mode, S in (("base", 1), ("store", 1), ("fixup", 1), ("switch", 2)):
        r = regs(K_.launcher(a, b, c, P, locks, mode=mode, segments=S), (K_MAIN, mode, S))
        print(f"  registers K={K_MAIN} {mode} S={S}: {fmt_regs(r)}")
    return ok


# ---------------------------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------------------------


def sweep_kstep():
    print("\n## kstep: baseline time vs K (one tile per program)\n")
    print("| K | k-steps | us |")
    print("|---|---|---|")
    xs, ys = [], []
    for Kdim in KSTEP_KS:
        a, b, c = inputs(Kdim)
        P, locks = K_.make_workspace()
        ln = K_.launcher(a, b, c, P, locks)
        t = time_us(ln)
        xs.append(Kdim // 64)
        ys.append(t)
        print(f"| {Kdim} | {Kdim // 64} | {t:.1f} |", flush=True)
    slope, icpt, res = fit(xs, ys)
    print(f"\nk-step = {slope:.3f} us, fixed = {icpt:.1f} us, max residual {res:.2f} us  "
          f"({fmt_regs(regs(ln, (KSTEP_KS[-1], "base", 1)))})")
    return slope


def show(name, xs, ys, kstep, xlabel, fit_min=0):
    """One table row; the fit uses the points with x >= fit_min."""
    fx = [x for x in xs if x >= fit_min]
    slope, icpt, res = fit(fx, [y for x, y in zip(xs, ys) if x >= fit_min])
    pts = ", ".join(f"{x}: {y:.1f}" for x, y in zip(xs, ys))
    first = ys[xs.index(1)] - ys[xs.index(0)]
    print(f"| {name} | {pts} | {first:.2f} ({first / kstep:.2f}) | {slope:.2f} | {slope / kstep:.2f} | "
          f"{res:.2f} |", flush=True)
    return first, slope


def name_of(v):
    """Short label for the partial-exchange variant: P layout, store / read cache."""
    lay = "lane" if v.get("p_layout", 0) else "row-major"
    sc = v.get("store_cache", ".wt") or "default"
    rc = v.get("read_cache", ".cv") or "default"
    return lay, sc, rc


def sweep_store(kstep, ws):
    a, b, c, P, locks = ws
    print(f"\n## store: r partials (256 KiB each) per storer, then release flag; K={K_MAIN}")
    print("(one partial per program is the stream-K case; r = 4 with 256 storers is 256 MiB and spills the MALL)\n")
    print("| storers | P layout | store cache | r: us | first partial: us (k-steps) | slope: us per partial | k-steps | max residual (us) |")
    print("|---|---|---|---|---|---|---|---|")
    out = {}
    variants = [dict(p_layout=0), dict(p_layout=1), dict(p_layout=1, store_cache="")]
    for v in variants:
        lay, sc, _ = name_of(v)
        for name, (mod, inv) in STORERS.items():
            xs, ys = [0, 1, 2, 4], []
            for r in xs:
                ys.append(time_us(K_.launcher(a, b, c, P, locks, mode="store", sel_mod=mod, sel_inv=inv,
                                              fin=(r, 0, 1), **v)))
            out[(name, lay, sc)] = show(f"{name} | {lay} | {sc}", xs, ys, kstep, "r")
    print(f"\n({fmt_regs(regs(K_.launcher(a, b, c, P, locks, mode="store"), (K_MAIN, "store", 1)))})")
    return out


def sweep_read(kstep, ws):
    a, b, c, P, locks = ws
    print(f"\n## read: owner reads n partials into its accumulator; K={K_MAIN}")
    print("(read-only: partials host-filled, flags preset. fits use n >= 1: n = 0 skips the owner's per-quadrant setup)\n")
    print("| case | P layout | read cache | n: us | first partial: us (k-steps) | slope: us per partial | k-steps | max residual (us) |")
    print("|---|---|---|---|---|---|---|---|")
    out = {}
    variants = [
        ("16 owners", 16, dict(p_layout=0)),
        ("16 owners", 16, dict(p_layout=1)),
        ("16 owners", 16, dict(p_layout=1, read_cache="")),
        ("16 owners, no flag polling", 16, dict(p_layout=1, poll=0)),
        ("256 owners", 1, dict(p_layout=0)),
        ("256 owners", 1, dict(p_layout=1)),
    ]
    for owners, mod, v in variants:
        lay, _, rc = name_of(v)
        P.fill_(1e-3)
        locks.fill_(1)
        xs, ys = [0, 1, 2, 4, 8, 12, 15], []
        for n in xs:
            ys.append(time_us(K_.launcher(a, b, c, P, locks, mode="fixup", sel_mod=mod, fin=(0, n, 1), **v)))
        out[(owners, lay, rc)] = show(f"read-only, {owners} | {lay} | {rc}", xs, ys, kstep, "n", fit_min=1)
    # Full fixup: groups of s, contributors publish (store + release), owner waits and reads s-1.
    for v in (dict(p_layout=0), dict(p_layout=1), dict(p_layout=1, store_cache="", read_cache="")):
        lay, sc, rc = name_of(v)
        locks.zero_()
        xs, ys = [], []
        for s_ in [1, 2, 4, 8, 16]:
            t = time_us(K_.launcher(a, b, c, P, locks, mode="fixup", sel_mod=s_, fin=(0, s_ - 1, 1),
                                    fin_u=(1, 0, 0), reset_flags=1, **v))
            assert int(locks.sum()) == 0, "flags not re-armed"
            xs.append(s_ - 1)
            ys.append(t)
        out[("fixup", lay, sc)] = show(f"fixup, groups of s (x = s-1 peers) | {lay} | {sc} / {rc}", xs, ys,
                                       kstep, "s-1", fit_min=1)
    print(f"\n({fmt_regs(regs(K_.launcher(a, b, c, P, locks, mode="fixup"), (K_MAIN, "fixup", 1)))})")
    return out


# name: (sw = (n_store, write_c) at a switch, release, launcher variant)
SWITCH_MODES = {
    "none": ((0, 0), 1, {}),
    "c": ((0, 1), 1, {}),
    "partial, row-major": ((1, 0), 1, dict(p_layout=0)),
    "partial": ((1, 0), 1, dict(p_layout=1)),
    "partial, relaxed flag": ((1, 0), 0, dict(p_layout=1)),
    "partial, default cache": ((1, 0), 1, dict(p_layout=1, store_cache="")),
}


def sweep_switch(kstep, ws):
    a, b, c, P, locks = ws
    print(f"\n## switch: each tile runs as S segments of {K_MAIN}/S; x = S-1 switches per program")
    print("(partial: fp32 .wt store + drain + release flag, lane layout unless noted)\n")
    print("| mode | switchers | S-1: us | first switch: us (k-steps) | slope: us per switch | k-steps | max residual (us) |")
    print("|---|---|---|---|---|---|---|")
    out = {}
    for switchers, (mod, inv) in [("256", (1, 0)), ("16", (16, 0))]:
        for mode, (sw, rel, v) in SWITCH_MODES.items():
            if switchers == "16" and mode == "none":
                continue
            xs, ys = [], []
            for S in [1, 2, 4, 8]:
                ys.append(time_us(K_.launcher(a, b, c, P, locks, mode="switch", segments=S, sel_mod=mod,
                                              sel_inv=inv, sw=sw, release=rel, **v)))
                xs.append(S - 1)
            out[(mode, switchers)] = show(f"{mode} | {switchers}", xs, ys, kstep, "S-1")
    for S in [1, 2, 4, 8]:
        r = regs(K_.launcher(a, b, c, P, locks, mode="switch", segments=S, p_layout=1), (K_MAIN, "switch", S))
        print(f"(S={S}: {fmt_regs(r)})")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", nargs="+", default=["check", "kstep", "store", "read", "switch"],
                   choices=["check", "kstep", "store", "read", "switch"])
    p.add_argument("--kstep-us", type=float, default=None, help="skip the kstep sweep, use this value")
    args = p.parse_args()

    if os.environ.get("TRITON_AMDGCNAS_PLUGIN"):
        sys.exit("unset TRITON_AMDGCNAS_PLUGIN: the amdgcnas peephole faults on these builds' spill code")
    print(f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', 'unset')} "
          f"M=N={M} K={K_MAIN} launches/point={N_LAUNCH}")
    if "check" in args.sweep:
        print("\n## check")
        if not check():
            sys.exit("correctness check failed")
    kstep = args.kstep_us
    if "kstep" in args.sweep:
        kstep = sweep_kstep()
    if kstep is None:
        sys.exit("need --kstep-us or the kstep sweep")
    a, b, c = inputs(K_MAIN)
    ws = (a, b, c, *K_.make_workspace())
    if "store" in args.sweep:
        sweep_store(kstep, ws)
    if "read" in args.sweep:
        sweep_read(kstep, ws)
    if "switch" in args.sweep:
        sweep_switch(kstep, ws)


if __name__ == "__main__":
    main()
