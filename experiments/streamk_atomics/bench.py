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
"""Atomic stream-K tile endings against today's partial store and read.

    HIP_VISIBLE_DEVICES=0 python bench.py                       # check, asm, main, pair (warm)
    HIP_VISIBLE_DEVICES=0 python bench.py --sweep main pair --cold

Each point runs `--launches` launches (plus one warm-up) and reports, over all active programs
and launches, the median and 90th percentile of the per-program ending time, from the stamp
before the ending to the stamp after its memory operations completed. "span" is the median over
launches of last end minus first start, and GB/s is the bytes all active programs move divided
by the span. --cold reads a 1 GB buffer before every launch, which evicts L2 and the MALL.
"""

import argparse
import os
import statistics

import kernel as K  # imports triton before torch
import torch

K_STEP_US = 1.21
ACTIVE = [1, 2, 4, 16, 64, 256]
WRITERS = [1, 2, 4]
PAIR_TILES = [1, 2, 4, 8, 16, 32, 64, 128]
FP32_TILE = K.TILE_ELEMS * 4
BF16_TILE = K.TILE_ELEMS * 2
# mode: (bytes moved per program, writer counts to sweep)
MAIN_MODES = {
    "store": (FP32_TILE, [1]),
    "load": (FP32_TILE, [1]),
    "owner": (FP32_TILE + BF16_TILE, [1]),
    "c_store": (BF16_TILE, [1]),
    "atom_lane": (FP32_TILE, WRITERS),
    "atom_coal": (FP32_TILE, WRITERS),
    "c_atom_acc": (BF16_TILE, WRITERS),
    "c_atom_coal": (BF16_TILE, WRITERS),
}

_FLUSH = None


def flush():
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.ones(256 * 2**20, device="cuda", dtype=torch.float32)
    _FLUSH.sum()


def fill_src(ws, seed=0):
    """Small integers, so sums of up to 4 tiles are exact in fp32 and bf16."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ws.src.copy_(torch.randint(-32, 32, ws.src.shape, device="cuda", generator=g).float())
    ws.P.copy_(torch.randint(-32, 32, ws.P.shape, device="cuda", generator=g).float())


def slot(buf, i):
    return buf[i * K.TILE_ELEMS:(i + 1) * K.TILE_ELEMS]


# ---------------------------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------------------------


def check(repeats=3):
    ws = K.Workspace()
    fill_src(ws)
    lane, coal = K.lane_index(), K.coal_index()
    src_tiles = [K.tile_from_slot(slot(ws.src, i), lane) for i in range(K.NUM_PROGRAMS)]
    p_tiles = [K.tile_from_slot(slot(ws.P, i), lane) for i in range(K.NUM_PROGRAMS)]
    ok = True

    def report(name, good):
        nonlocal ok
        ok &= bool(good)
        print(f"- {name}: {'ok' if good else 'FAIL'}", flush=True)

    def tile_sum(t, w):
        return sum(src_tiles[t * w + j] for j in range(w))

    A = K.NUM_PROGRAMS
    P0 = ws.P.clone()
    ws.P.zero_()
    K.launch(ws, "store", A, 1)
    torch.cuda.synchronize()
    report("store: P[i] == src[i]", torch.equal(ws.P, ws.src))
    ws.P.copy_(P0)

    ws.W.zero_()
    K.launch(ws, "load", A, 1, sink=1)
    torch.cuda.synchronize()
    report("load: src[i] + P[i] (sink store) is right", torch.equal(ws.W, ws.src + ws.P))

    ws.C.zero_()
    K.launch(ws, "c_store", A, 1)
    torch.cuda.synchronize()
    report("c_store: C tile == src tile",
           all(torch.equal(K.c_tile(ws.C, t).float(), src_tiles[t]) for t in range(A)))

    ws.C.zero_()
    K.launch(ws, "owner", A, 1)
    torch.cuda.synchronize()
    report("owner: C tile == src tile + P tile",
           all(torch.equal(K.c_tile(ws.C, t).float(), src_tiles[t] + p_tiles[t]) for t in range(A)))

    for w in WRITERS:
        tiles = A // w
        for mode, index in (("atom_lane", lane), ("atom_coal", coal)):
            good = True
            for _ in range(repeats):
                ws.W.zero_()
                K.launch(ws, mode, A, w)
                torch.cuda.synchronize()
                good &= all(torch.equal(K.tile_from_slot(slot(ws.W, t), index), tile_sum(t, w)) for t in range(tiles))
            report(f"{mode}, {w} writers per tile, x{repeats}: W[t] == sum of the writers' tiles", good)
        for mode in ("c_atom_acc", "c_atom_coal"):
            good = True
            for _ in range(repeats):
                ws.C.zero_()
                K.launch(ws, mode, A, w)
                torch.cuda.synchronize()
                good &= all(torch.equal(K.c_tile(ws.C, t).float(), tile_sum(t, w)) for t in range(tiles))
            report(f"{mode}, {w} writers per tile, x{repeats}: C tile == sum (bf16)", good)
        if w > 1:
            _, xcc = K.records(ws, A)
            spread = sum(len({xcc[t * w + j] for j in range(w)}) == w for t in range(tiles))
            report(f"{w} writers per tile run on {w} different XCDs for {spread}/{tiles} tiles", spread == tiles)

    good = True
    for _ in range(repeats):
        ws.C.zero_()
        K.launch(ws, "pair", A, 2)
        torch.cuda.synchronize()
        good &= all(torch.equal(K.c_tile(ws.C, t).float(), tile_sum(t, 2)) for t in range(A // 2))
    report(f"pair x{repeats}: C tile == contributor's bf16 store + owner's pk_add", good)

    good = True
    for _ in range(repeats):
        ws.C.zero_()
        ws.P.copy_(P0)
        K.launch(ws, "pair_today", A, 2)
        torch.cuda.synchronize()
        good &= all(torch.equal(K.c_tile(ws.C, t).float(), tile_sum(t, 2)) for t in range(A // 2))
    report(f"pair_today x{repeats}: C tile == owner's tile + contributor's fp32 partial", good)
    return ok


# ---------------------------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------------------------

ASM_OPS = ["buffer_atomic_add_f32", "buffer_atomic_pk_add_bf16", "buffer_store_dwordx4", "buffer_store_dwordx2",
           "buffer_store_dword ", "buffer_load_dwordx4", "ds_write", "ds_read"]


def asm_table():
    ws = K.Workspace()
    print("Per-thread instruction counts in the kernel (all four quadrants, including the 4 x 16 "
          "buffer_load_dwordx4 of the synthetic accumulator). 'returns' counts atomics with sc0 "
          "(the returning form).\n")
    print("| mode | VGPRs | spills | " + " | ".join(o.strip() for o in ASM_OPS) + " | returns |")
    print("|---|---|---|" + "---|" * (len(ASM_OPS) + 1))
    samples = {}
    for mode in K.MODES:
        # Run-time arguments equal to 1 are specialised as constants, so use 2 for both.
        k = K.launch(ws, mode, 2, 2)
        torch.cuda.synchronize()
        lines = [ln.strip() for ln in k.asm["amdgcn"].splitlines()]
        counts = [sum(ln.startswith(o) for ln in lines) for o in ASM_OPS]
        atomics = [ln.split(";")[0].strip() for ln in lines if ln.startswith("buffer_atomic")]
        returns = sum("sc0" in ln for ln in atomics)
        if atomics:
            samples.setdefault(atomics[0].split()[0], atomics[0])
        print(f"| {mode} | {k.n_regs} | {k.n_spills} | " + " | ".join(map(str, counts)) + f" | {returns} |")
    print("\nSample atomics:\n")
    for s in samples.values():
        print(f"    {s}")


# ---------------------------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------------------------


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def run(ws, mode, A, w, launches, cold):
    """Per launch: the (A, 3) stamp matrix in us, relative to that launch's first start."""
    out = []
    for i in range(launches + 1):
        if cold:
            flush()
        K.launch(ws, mode, A, w)
        torch.cuda.synchronize()
        if i:
            out.append(K.records(ws, A)[0])
    return out


def ks(us):
    return us / K_STEP_US


def sweep_main(ws, launches, cold):
    print(f"\n## main: one ending per active program ({'cold' if cold else 'warm'} caches)\n")
    print("| mode | writers per tile | active | KiB per program | median us | p90 us | median k-steps | "
          "span us | GB/s | start skew us |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for mode, (nbytes, writers) in MAIN_MODES.items():
        for w in writers:
            for A in ACTIVE:
                if A < w:
                    continue
                recs = run(ws, mode, A, w, launches, cold)
                d = [float(x) for r in recs for x in (r[:, 2] - r[:, 0])]
                span = statistics.median(float(r[:, 2].max()) for r in recs)
                skew = statistics.median(float(r[:, 0].max()) for r in recs)
                med = statistics.median(d)
                gbs = A * nbytes / (span * 1e-6) / 1e9
                print(f"| {mode} | {w} | {A} | {nbytes // 1024} | {med:.2f} | {pct(d, 0.9):.2f} | {ks(med):.2f} | "
                      f"{span:.2f} | {gbs:.0f} | {skew:.2f} |", flush=True)


def pair_phases(recs, S):
    """Medians of: contributor publish, owner wait, owner ending, and the tile's whole handshake
    (owner's end minus the pair's first start)."""
    med = statistics.median
    return (
        med(float(r[i, 1] - r[i, 0]) for r in recs for i in range(0, 2 * S, 2)),
        med(float(r[i, 1] - r[i, 0]) for r in recs for i in range(1, 2 * S, 2)),
        med(float(r[i, 2] - r[i, 1]) for r in recs for i in range(1, 2 * S, 2)),
        med(float(r[i, 2] - min(r[i, 0], r[i - 1, 0])) for r in recs for i in range(1, 2 * S, 2)),
    )


def sweep_pair(ws, launches, cold):
    print(f"\n## pair: S tiles, one contributor and one owner each ({'cold' if cold else 'warm'} caches)\n")
    print("today (pair_today): the contributor stores its fp32 partial (lane-contiguous), drains and raises a "
          "release flag; the owner polls, acquires, reads the partial, adds and stores bf16 C.")
    print("atomic (pair): the contributor stores its partial as bf16 into C, drains and raises the flag; the "
          "owner polls, acquires and pk_adds its own tile into C.")
    print("Both start with the accumulator ready on both programs, so 'whole' is the fixup's critical path "
          "when neither program has MAC work left. us (k-steps).\n")
    print("| S | today: publish | today: wait | today: owner ending | today: whole | "
          "atomic: publish | atomic: wait | atomic: owner ending | atomic: whole |")
    print("|---|---|---|---|---|---|---|---|---|")

    def f(us):
        return f"{us:.2f} ({ks(us):.2f})"

    for S in PAIR_TILES:
        today = pair_phases(run(ws, "pair_today", 2 * S, 2, launches, cold), S)
        atom = pair_phases(run(ws, "pair", 2 * S, 2, launches, cold), S)
        print(f"| {S} | " + " | ".join(f(x) for x in today + atom) + " |", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", nargs="+", default=["check", "asm", "main", "pair"],
                   choices=["check", "asm", "main", "pair"])
    p.add_argument("--launches", type=int, default=20)
    p.add_argument("--cold", action="store_true")
    args = p.parse_args()
    print(f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', 'unset')} launches/point={args.launches} "
          f"k-step={K_STEP_US} us")
    if "check" in args.sweep:
        print("\n## check\n")
        if not check():
            raise SystemExit("correctness check failed")
    if "asm" in args.sweep:
        print("\n## asm\n")
        asm_table()
    ws = K.Workspace()
    fill_src(ws)
    if "main" in args.sweep:
        sweep_main(ws, args.launches, args.cold)
    if "pair" in args.sweep:
        sweep_pair(ws, args.launches, args.cold)


if __name__ == "__main__":
    main()
