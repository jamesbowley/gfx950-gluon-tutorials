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
"""Why the reduce-scatter slice takes 8-11 us in the GEMM when the probe reads the same data in
3-5 us.

    cd kernels/gemm/intra_wave/a16w16
    HIP_VISIBLE_DEVICES=1 python v21_streamk_l2_partials/slice_timing.py

Builds two copies of matmul_kernel.py with fixup_timing.py's stamps plus:
- one more stamp inside reduce_scatter_slice, after its reads have completed and before the
  bias and the C store, so the slice splits into "reads" and "bias + C store";
- (second copy) the slice replaced by the streamk_l2_partials probe's loop: contiguous 4 KiB
  blocks, the n partials summed one after another, no bias, no C store. C is not written, so
  that copy is for timing only (its "wrong" column is expected to be every launch).

Also reports, for every split program, how many other split programs were still in their MAC
loop when its slice started (median), to see whether the slice overlaps other programs' A/B
streaming. Medians in us over the split programs and launches.
"""

import argparse
import importlib.util
import os
import statistics
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import bench  # noqa: E402  (plugin hooks and the path to common.py; imports triton before torch)
import torch  # noqa: E402

import fixup_timing  # noqa: E402

OFFSET = fixup_timing.OFFSET
CASES = {"4352x4096x8192": [8], "3072x5632x8192": [16], "4096x4608x8192": [4]}

PROBE_SLICE = '''

@gluon.jit
def probe_slice(p_ptr, bias_ptr, c_tile, b, j, n, pid_n, stride_cm, stride_cn, BLOCK_N: gl.constexpr,
                ADD_BIAS: gl.constexpr, NB: gl.constexpr, U: gl.constexpr, out_dtype: gl.constexpr,
                LOAD_CACHE: gl.constexpr, sync_ptr, spid_):
    """The streamk_l2_partials probe's slice loop (timing only): contiguous 4 KiB blocks, the n
    partials summed one after another, no bias and no C store."""
    L2D: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [1, 4], [1, 0])
    SLOT: gl.constexpr = 4 * 128 * 128
    i = gl.arange(0, NB, gl.SliceLayout(1, L2D))[:, None]
    e = gl.arange(0, 1024, gl.SliceLayout(0, L2D))[None, :]
    lo = (64 * j) // n
    hi = (64 * (j + 1)) // n
    tot = gl.zeros((NB, 1024), gl.float32, L2D)
    for b0 in range(lo, hi, NB):
        blk = b0 + i
        valid = (blk < hi) & (e >= 0)
        off = blk * 1024 + e
        for m in range(n):
            tot += gl.amd.cdna3.buffer_load(ptr=p_ptr + (b + m).to(gl.int64) * SLOT, offsets=off, mask=valid,
                                            other=0.0, cache=LOAD_CACHE)
    _stamp(sync_ptr, spid_, 5, True)
    zero = gl.zeros([1], gl.int32, gl.BlockedLayout([1], [64], [gl.num_warps()], [0]))
    gl.store(sync_ptr + 1280 + spid_ * 8 + 6 + zero, gl.sum(gl.sum(tot, axis=1), axis=0).to(gl.int32) + zero)
'''


def patch(src, probe):
    src = fixup_timing.patch(src)

    def rep(old, new):
        nonlocal src
        assert src.count(old) == 1, old
        src = src.replace(old, new)

    rep("    out_dtype: gl.constexpr,\n    LOAD_CACHE: gl.constexpr,\n):\n    \"\"\"Program j's 1/n",
        "    out_dtype: gl.constexpr,\n    LOAD_CACHE: gl.constexpr,\n    sync_ptr,\n    spid_,\n):\n"
        "    \"\"\"Program j's 1/n")
    rep("                )\n        # Block q * 16 + v: quadrant q",
        "                )\n        _stamp(sync_ptr, spid_, 5, True)\n        # Block q * 16 + v: quadrant q")
    rep("out_dtype, RS_LOAD,\n            )\n            _stamp(sync_ptr, spid, 4, True)",
        "out_dtype, RS_LOAD, sync_ptr, spid,\n            )\n            _stamp(sync_ptr, spid, 4, True)")
    if probe:
        rep("\n\n@gluon.jit\ndef reduce_scatter_slice(", PROBE_SLICE + "\n\n@gluon.jit\ndef reduce_scatter_slice(")
        rep("            reduce_scatter_slice(\n                p_ptr, bias_ptr, c_tl_base",
            "            probe_slice(\n                p_ptr, bias_ptr, c_tl_base")
    return src


def load(probe):
    src = open(os.path.join(HERE, "matmul_kernel.py")).read()
    d = tempfile.mkdtemp(prefix="v21_slice_")
    name = "v21_slice_probe" if probe else "v21_slice_real"
    path = os.path.join(d, name + ".py")
    with open(path, "w") as f:
        f.write(patch(src, probe))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def measure(mod, M, N, K, n, mode, launches):
    S = (M // 256) * (N // 256) % 256
    os.environ.update(STREAMK_POLICY="one_tile", STREAMK_FIXUP="rs", STREAMK_PARTIALS=mode,
                      STREAMK_NUM_PROGRAMS=str(S * n))
    a = torch.rand((M, K), device=bench.DEVICE, dtype=torch.bfloat16) - 0.5
    b = torch.rand((N, K), device=bench.DEVICE, dtype=torch.bfloat16).T - 0.5
    c = torch.empty((M, N), device=bench.DEVICE, dtype=torch.bfloat16)
    P, locks, tile_sync, epochs = mod.streamk_workspace(a.device, 256, 256, 256)
    if tile_sync.numel() < OFFSET + 256 * 8:
        bigger = torch.zeros(OFFSET + 256 * 8, device=a.device, dtype=torch.int32)
        mod._WORKSPACE[(str(a.device), 256, 256, 256)] = (P, locks, bigger, epochs)
        tile_sync = bigger
    ref = torch.matmul(a, b)
    ph = {k: [] for k in ("mac", "store", "barrier", "reads", "bias_c", "slice", "in_mac")}
    wrong = 0
    for i in range(launches + 1):
        tile_sync[OFFSET:].zero_()
        mod.matmul(a, b, c)
        torch.cuda.synchronize()
        wrong += not torch.allclose(c, ref, atol=1e-1, rtol=1e-2)
        if i == 0:
            continue
        st = tile_sync[OFFSET:OFFSET + 256 * 8].view(256, 8)[:, :6].to(torch.int64).cpu().tolist()
        rows = [r for r in st if r[0] and r[2] and r[3] and r[4] and r[5]]
        t0 = min(r[0] for r in rows)

        def t(x):
            return ((x - t0) % 2**32) * 0.01

        mac_end = [t(r[1]) for r in rows]
        for r in rows:
            ph["mac"].append(t(r[1]) - t(r[0]))
            ph["store"].append(t(r[2]) - t(r[1]))
            ph["barrier"].append(t(r[3]) - t(r[2]))
            ph["reads"].append(t(r[5]) - t(r[3]))
            ph["bias_c"].append(max(t(r[4]) - t(r[5]), 0.0))
            ph["slice"].append(t(r[4]) - t(r[3]))
            ph["in_mac"].append(sum(e > t(r[3]) for e in mac_end))
    out = {k: statistics.median(v) for k, v in ph.items()}
    out["wrong"] = wrong
    out["programs"] = len(rows)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modes", nargs="+", default=["mem", "l2"], choices=["mem", "wt", "l2"])
    p.add_argument("--launches", type=int, default=10)
    args = p.parse_args()
    mods = {"GEMM slice": load(False), "probe loop": load(True)}
    print("Medians in us over the split programs and launches. reads: from the barrier to the slice's reads "
          "completing; bias + C: the rest of the slice; in MAC: other split programs still in their MAC loop "
          "when this program's slice starts. The probe loop writes no C, so its results are wrong by design.\n")
    print("| shape | S | n | partials | slice code | MAC | store | barrier | reads | bias + C | slice | in MAC "
          "(of programs) | wrong |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s, ns in CASES.items():
        M, N, K = (int(x) for x in s.split("x"))
        S = (M // 256) * (N // 256) % 256
        for n in ns:
            for mode in args.modes:
                for name, mod in mods.items():
                    r = measure(mod, M, N, K, n, mode, args.launches)
                    print(f"| {s} | {S} | {n} | {mode} | {name} | {r['mac']:.1f} | {r['store']:.1f} | "
                          f"{r['barrier']:.1f} | {r['reads']:.1f} | {r['bias_c']:.1f} | {r['slice']:.1f} | "
                          f"{r['in_mac']:.0f} (of {r['programs']}) | {r['wrong']} |", flush=True)


if __name__ == "__main__":
    main()
