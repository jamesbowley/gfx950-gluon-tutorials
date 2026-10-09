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
"""Where the one-tile reduce-scatter tail's time goes, per STREAMK_PARTIALS mode and split count.

    cd kernels/gemm/intra_wave/a16w16
    HIP_VISIBLE_DEVICES=3 python v21_streamk_l2_partials/fixup_timing.py
    HIP_VISIBLE_DEVICES=3 python v21_streamk_l2_partials/fixup_timing.py --cases 4352x4096x8192:8,16

Builds a copy of matmul_kernel.py with s_memrealtime stamps (100 MHz, one clock for the whole
device) patched in, each after a workgroup barrier and, where it ends a memory phase, after
s_waitcnt vmcnt(0):

- start: the program begins its stream-K segment (persistent loop done and drained);
- mac: the segment's MAC loop is done;
- stored: the partial stores have completed;
- synced: the tile barrier is passed;
- done: the slice's C stores have completed.

Every case runs STREAMK_POLICY=one_tile with the reduce-scatter ending and n programs per
tile (STREAMK_NUM_PROGRAMS = S n). Medians over the split programs and launches; the tail is
the last done minus the first start. "kernel" is do_bench (which zeroes 256 MB before every
launch) of the unpatched v21 in the same mode; "kernel, clean" is the median of CUDA-event
times with a 256 MB read (clean, cold caches) before every launch. One k-step is about 1.21 us.
"""

import argparse
import importlib.util
import os
import statistics
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import bench  # noqa: E402  (plugin hooks and the path to common.py; imports triton before torch)
import torch  # noqa: E402
import triton  # noqa: E402

from v21_streamk_l2_partials import matmul_kernel as plain  # noqa: E402

K_STEP_US = 1.21
OFFSET = 1280  # first stamp slot in tile_sync (int32), after the barriers, L2 flags and XCC records
CASES = {
    "3072x5632x8192": [4, 8, 16, 32],
    "4352x4096x8192": [4, 8, 16],
    "4096x4608x8192": [4, 8],
    "3328x5120x8192": [16, 32],
}

STAMP_FN = '''

@gluon.jit
def _stamp(sync_ptr, spid, i: gl.constexpr, WAIT: gl.constexpr):
    if WAIT:
        pre: gl.constexpr = "s_waitcnt vmcnt(0)\\ns_barrier\\n"
    else:
        pre: gl.constexpr = "s_barrier\\n"
    t = gl.inline_asm_elementwise(
        pre + "s_memrealtime $0\\ns_waitcnt lgkmcnt(0)", "=s,v,~{memory}", [_one()],
        dtype=gl.int64, is_pure=False, pack=1,
    )
    zero = gl.zeros([1], gl.int32, gl.BlockedLayout([1], [64], [gl.num_warps()], [0]))
    gl.store(sync_ptr + 1280 + spid * 8 + i + zero, t.to(gl.int32))
'''


def patch(src):
    def rep(old, new):
        nonlocal src
        assert src.count(old) == 1, old
        src = src.replace(old, new)

    rep("\n\n@gluon.jit\ndef drain_and_barrier():", STAMP_FN + "\n\n@gluon.jit\ndef drain_and_barrier():")
    rep("        if active:\n", "        if active:\n            _stamp(sync_ptr, spid, 0, True)\n")
    rep(
        "    drain_and_barrier()\n\n    gStoreLayoutC: gl.constexpr",
        "    drain_and_barrier()\n    _stamp(sync_ptr, spid, 1, False)\n\n    gStoreLayoutC: gl.constexpr",
    )
    rep(
        "            drain_and_barrier()\n            tile_barrier(sync_ptr, spid - j, n_tile, spid, epoch, NUM_PROGRAMS, PARTIALS)\n",
        "            drain_and_barrier()\n            _stamp(sync_ptr, spid, 2, False)\n"
        "            tile_barrier(sync_ptr, spid - j, n_tile, spid, epoch, NUM_PROGRAMS, PARTIALS)\n"
        "            _stamp(sync_ptr, spid, 3, False)\n",
    )
    rep(
        "                pid_n, stride_cm, stride_cn, BLOCK_N, ADD_BIAS, RS_NB, RS_U, out_dtype, RS_LOAD,\n            )\n",
        "                pid_n, stride_cm, stride_cn, BLOCK_N, ADD_BIAS, RS_NB, RS_U, out_dtype, RS_LOAD,\n            )\n"
        "            _stamp(sync_ptr, spid, 4, True)\n",
    )
    return src


def load_patched(kernel=None):
    src = open(kernel or os.path.join(HERE, "matmul_kernel.py")).read()
    d = tempfile.mkdtemp(prefix="v21_timing_")
    path = os.path.join(d, "v21_timed_kernel.py")
    with open(path, "w") as f:
        f.write(patch(src))
    spec = importlib.util.spec_from_file_location("v21_timed_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def clean_us(a, b, c, launches):
    big = torch.empty(256 * 2**20 // 4, device=a.device, dtype=torch.int32)
    ts = []
    for i in range(launches + 1):
        big.sum()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        plain.matmul(a, b, c)
        e.record()
        torch.cuda.synchronize()
        if i:
            ts.append(s.elapsed_time(e) * 1e3)
    return median(ts)


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
    ph = {k: [] for k in ("mac", "store", "sync", "slice", "ending", "tail")}
    wrong = 0
    for i in range(launches + 1):
        tile_sync[OFFSET:].zero_()
        mod.matmul(a, b, c)
        torch.cuda.synchronize()
        wrong += not torch.allclose(c, ref, atol=1e-1, rtol=1e-2)
        if i == 0:
            continue
        st = tile_sync[OFFSET:OFFSET + 256 * 8].view(256, 8)[:, :5].to(torch.int64).cpu().tolist()
        rows = [r for r in st if r[0] and r[2] and r[3] and r[4]]
        t0 = min(r[0] for r in rows)

        def t(x):
            return ((x - t0) % 2**32) * 0.01

        for r in rows:
            ph["mac"].append(t(r[1]) - t(r[0]))
            ph["store"].append(t(r[2]) - t(r[1]))
            ph["sync"].append(t(r[3]) - t(r[2]))
            ph["slice"].append(t(r[4]) - t(r[3]))
            ph["ending"].append(t(r[4]) - t(r[1]))
        ph["tail"].append(max(t(r[4]) for r in rows))
    out = {k: median(v) for k, v in ph.items()}
    ms = triton.testing.do_bench(lambda: plain.matmul(a, b, c), quantiles=[0.5])
    out["kernel"] = (ms[0] if isinstance(ms, (list, tuple)) else ms) * 1e3
    out["clean"] = clean_us(a, b, c, launches)
    out["wrong"] = wrong
    n_eff = plain.xcd_local_split(S, n, K // 128, 256, 8) if mode == "l2" else n
    out["n"] = n_eff
    out["mib"] = -(-S // 8) * n_eff * 0.25
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="*", help="MxNxK:n1,n2,... (default: the plan's shapes and splits)")
    parser.add_argument("--modes", nargs="+", default=["mem", "wt", "l2"], choices=["mem", "wt", "l2"])
    parser.add_argument("--launches", type=int, default=10)
    parser.add_argument("--kernel", help="time this copy of matmul_kernel.py instead (an experiment)")
    args = parser.parse_args()
    cases = CASES
    if args.cases:
        cases = {s.split(":")[0]: [int(x) for x in s.split(":")[1].split(",")] for s in args.cases}
    mod = load_patched(args.kernel)
    print("Medians in us (k-steps of 1.21 us in brackets) over the split programs and launches. MiB: the "
          "busiest XCD's partials (l2's XCD-local split), against 4 MiB of L2.\n")
    print("| shape | S | n | mode | MiB per XCD | MAC | store | barrier | slice | ending | tail | kernel (do_bench) "
          "| kernel (clean) | wrong |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    def f(us):
        return f"{us:.1f} ({us / K_STEP_US:.1f})"

    for s, ns in cases.items():
        M, N, K = (int(x) for x in s.split("x"))
        S = (M // 256) * (N // 256) % 256
        for n in ns:
            for mode in args.modes:
                r = measure(mod, M, N, K, n, mode, args.launches)
                print(f"| {s} | {S} | {r['n']} | {mode} | {r['mib']:g} | {f(r['mac'])} | {f(r['store'])} | "
                      f"{f(r['sync'])} | {f(r['slice'])} | {f(r['ending'])} | {f(r['tail'])} | {r['kernel']:.1f} | "
                      f"{r['clean']:.1f} | {r['wrong']} |", flush=True)


if __name__ == "__main__":
    main()
