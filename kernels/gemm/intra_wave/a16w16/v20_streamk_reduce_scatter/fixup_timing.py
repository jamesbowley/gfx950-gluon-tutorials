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
"""Where the one-tile stream-K tail's time goes, for both endings (STREAMK_FIXUP=rs and owner).

    cd kernels/gemm/intra_wave/a16w16
    HIP_VISIBLE_DEVICES=2 python v20_streamk_reduce_scatter/fixup_timing.py
    HIP_VISIBLE_DEVICES=2 python v20_streamk_reduce_scatter/fixup_timing.py --shapes 3328x5120x8192

Builds a copy of matmul_kernel.py with s_memrealtime stamps (100 MHz, one clock for the whole
device) patched in, each after a workgroup barrier and, where it ends a memory phase, after
s_waitcnt vmcnt(0):

- start: the stream-K phase begins (persistent loop done and drained);
- mac: the segment's MAC loop is done;
- stored: (rs) the partial stores have completed; (owner) same as mac;
- synced: (rs) the tile barrier is passed; (owner) the owner has seen every peer's flag, a
  contributor has published;
- done: the C stores (rs: the slice; owner: the whole tile on the owner) have completed.

The stamps go after the barrier counts in the tile_sync buffer, which the harness allocates
larger. Each stamp costs a barrier and a memory drain, so absolute times are a little longer
than the real kernel's. Per shape, the table gives medians over the split programs (rs) or over
the owners (owner) of each phase, the tail (last done minus first start) and the kernel's
do_bench time. One k-step is about 1.21 us.
"""

import argparse
import importlib.util
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import bench  # noqa: E402  (plugin hooks and the path to common.py)
import torch  # noqa: E402
import triton  # noqa: E402

K_STEP_US = 1.21
STAMPS = ["start", "mac", "stored", "synced", "done"]
OFFSET = 1024  # first stamp slot in tile_sync (int32), after 2 x 256 barrier words

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
    gl.store(sync_ptr + 1024 + spid * 8 + i + zero, t.to(gl.int32))
'''


def patch(src):
    def rep(old, new):
        nonlocal src
        assert src.count(old) == 1, old
        src = src.replace(old, new)

    rep("\n\n@gluon.jit\ndef drain_and_barrier():", STAMP_FN + "\n\n@gluon.jit\ndef drain_and_barrier():")
    rep(
        "    drain_and_barrier()\n\n    gStoreLayoutC: gl.constexpr",
        "    drain_and_barrier()\n    _stamp(sync_ptr, spid, 1, False)\n\n    gStoreLayoutC: gl.constexpr",
    )
    rep(
        "            drain_and_barrier()\n            tile_barrier(sync_ptr, spid - j, n_tile)\n",
        "            drain_and_barrier()\n            _stamp(sync_ptr, spid, 2, False)\n"
        "            tile_barrier(sync_ptr, spid - j, n_tile)\n            _stamp(sync_ptr, spid, 3, False)\n",
    )
    rep(
        "                pid_n, stride_cm, stride_cn, BLOCK_N, ADD_BIAS, RS_NB, RS_U, out_dtype,\n            )\n",
        "                pid_n, stride_cm, stride_cn, BLOCK_N, ADD_BIAS, RS_NB, RS_U, out_dtype,\n            )\n"
        "            _stamp(sync_ptr, spid, 4, True)\n",
    )
    rep(
        "        if n_peers > 0:\n            gl.atomic_add(locks_ptr + spid, 0, sem=\"acquire\", scope=\"gpu\")\n",
        "        if n_peers > 0:\n            gl.atomic_add(locks_ptr + spid, 0, sem=\"acquire\", scope=\"gpu\")\n"
        "        _stamp(sync_ptr, spid, 2, False)\n",
    )
    rep(
        "            gl.atomic_xchg(locks_ptr + spid, 1, sem=\"release\", scope=\"gpu\")\n",
        "            gl.atomic_xchg(locks_ptr + spid, 1, sem=\"release\", scope=\"gpu\")\n"
        "        _stamp(sync_ptr, spid, 3, False)\n",
    )
    rep(
        "            gl.atomic_xchg(locks_ptr + spid + PEER_STEP * (1 + pj), 0, sem=\"relaxed\", scope=\"gpu\")\n\n\n",
        "            gl.atomic_xchg(locks_ptr + spid + PEER_STEP * (1 + pj), 0, sem=\"relaxed\", scope=\"gpu\")\n"
        "        _stamp(sync_ptr, spid, 4, True)\n\n\n",
    )
    rep(
        "    gl.amd.cdna4.async_copy.wait_group(0)\n\n    if STREAMK_TILES > NUM_PROGRAMS:",
        "    gl.amd.cdna4.async_copy.wait_group(0)\n"
        "    _stamp(sync_ptr, get_logical_chiplet_mapped_pids(NUM_PROGRAMS, NUM_XCDS), 0, True)\n\n"
        "    if STREAMK_TILES > NUM_PROGRAMS:",
    )
    return src


def load_patched():
    src = open(os.path.join(HERE, "matmul_kernel.py")).read()
    d = tempfile.mkdtemp(prefix="v20_timing_")
    path = os.path.join(d, "v20_timed_kernel.py")
    with open(path, "w") as f:
        f.write(patch(src))
    spec = importlib.util.spec_from_file_location("v20_timed_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def split_counts(M, N, K):
    """Programs per one-tile stream-K tile, per spid (0 for idle spids), as in the kernel."""
    s = (M // 256) * (N // 256) % 256
    pairs = K // 128
    n_lo, n_hi = min(256 // s, pairs), min(256 // s + 1, pairs)
    extra = 256 % s if n_hi > n_lo else 0
    out = []
    for t in range(s):
        n = n_hi if t < extra else n_lo
        out += [(n, j) for j in range(n)]
    return out + [(0, 0)] * (256 - len(out))


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def measure(mod, M, N, K, fixup, launches):
    os.environ["STREAMK_FIXUP"] = fixup
    os.environ["STREAMK_POLICY"] = "one_tile"
    a = torch.rand((M, K), device=bench.DEVICE, dtype=torch.bfloat16) - 0.5
    b = torch.rand((N, K), device=bench.DEVICE, dtype=torch.bfloat16).T - 0.5
    P, locks, tile_sync = mod.streamk_workspace(a.device, 256, 256, 256)
    if tile_sync.numel() < OFFSET + 256 * 8:
        bigger = torch.zeros(OFFSET + 256 * 8, device=a.device, dtype=torch.int32)
        mod._WORKSPACE[(str(a.device), 256, 256, 256)] = (P, locks, bigger)
        tile_sync = bigger
    ref = torch.matmul(a, b)
    rows = []
    for _ in range(launches):
        c = mod.matmul(a, b)
        torch.cuda.synchronize()
        assert torch.allclose(c, ref, atol=1e-1, rtol=1e-2)
        st = tile_sync[OFFSET:].view(256, 8)[:, :5].to(torch.int64).cpu()
        rows.append(st)
    counts = split_counts(M, N, K)
    phases = {k: [] for k in ("mac", "store", "sync", "fixup", "tail")}
    for st in rows[1:]:
        t0 = min(int(st[p, 0]) for p in range(256))
        rel = [[(int(st[p, i]) - t0) % 2**32 * 0.01 for i in range(5)] for p in range(256)]
        last_done = 0.0
        for p, (n, j) in enumerate(counts):
            if n < 2:
                continue
            r = rel[p]
            last_done = max(last_done, r[4])
            if fixup == "owner" and j != 0:
                continue
            phases["mac"].append(r[1] - r[0])
            phases["store"].append((r[2] - r[1]) if fixup == "rs" else 0.0)
            phases["sync"].append(r[3] - r[2] if fixup == "rs" else r[2] - r[1])
            phases["fixup"].append(r[4] - r[3] if fixup == "rs" else r[4] - r[2])
        phases["tail"].append(last_done)
    ms = triton.testing.do_bench(lambda: mod.matmul(a, b), quantiles=[0.5])
    ms = ms[0] if isinstance(ms, (list, tuple)) else ms
    return {k: median(v) for k, v in phases.items()}, ms * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--shapes", nargs="*",
        default=["3328x5120x8192", "4352x4096x4096", "4352x4096x8192", "4352x4352x8192", "4096x5120x8192",
                 "4096x6144x8192"],
    )
    parser.add_argument("--launches", type=int, default=6)
    args = parser.parse_args()
    mod = load_patched()
    print("Medians in us (k-steps of 1.21 us in brackets). rs: over all split programs; owner: over the")
    print("owners, whose 'sync' is the wait for the last peer flag and 'fixup' the reads plus the C store.\n")
    print("| shape | S | n | fixup | MAC | partial store | barrier / flag wait | slice or owner fixup | tail | kernel (us) |")
    print("|---|---|---|---|---|---|---|---|---|---|")

    def f(us):
        return f"{us:.1f} ({us / K_STEP_US:.1f})"

    for s in args.shapes:
        M, N, K = (int(x) for x in s.split("x"))
        S = (M // 256) * (N // 256) % 256
        ns = sorted({n for n, _ in split_counts(M, N, K) if n > 1})
        for fixup in ("owner", "rs"):
            ph, us = measure(mod, M, N, K, fixup, args.launches)
            print(
                f"| {s} | {S} | {'/'.join(map(str, ns))} | {fixup} | {f(ph['mac'])} | {f(ph['store'])} | "
                f"{f(ph['sync'])} | {f(ph['fixup'])} | {f(ph['tail'])} | {us:.1f} |",
                flush=True,
            )


if __name__ == "__main__":
    main()
