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
"""Where the one-tile stream-K tail's time goes: v23's endings, v20's and v22's atomic ending.

    cd kernels/gemm/intra_wave/a16w16
    HIP_VISIBLE_DEVICES=1 python v23_streamk_rs_band/fixup_timing.py --tile-limit           # E2 cells and configs
    HIP_VISIBLE_DEVICES=1 python v23_streamk_rs_band/fixup_timing.py --tile-limit --cases 16:8192:8 \
        --configs 20:rs 23:rs 23:rs+STREAMK_PARTIALS=bf16 22:atomic_iz
    HIP_VISIBLE_DEVICES=2 python v23_streamk_rs_band/fixup_timing.py --shapes 3328x5120x8192
    HIP_VISIBLE_DEVICES=2 python v23_streamk_rs_band/fixup_timing.py --two-tile --shapes 5632x3072x6144

v20's fixup_timing.py, extended. A configuration is VERSION:FIXUP[+NAME=VALUE...]: the version
(20, 22 or 23) whose matmul_kernel.py is patched, its STREAMK_FIXUP (rs, owner, and for v22 atomic
or atomic_iz) and further STREAMK_* variables (STREAMK_PARTIALS, STREAMK_RS_READERS).

--two-tile times the two-tile path's head piece, partial publish (stores, then the flag
atomic) and the owner's wait and fixup instead, after some work before every launch
(--modes; read or zero 256 MB, optionally re-reading A and B), with the kernel's relaxed flag
and a release-flag control (--flags; release adds the compiler's buffer_wbl2, as v20 had
before 2026-10-08).

Builds a copy of the version's matmul_kernel.py with s_memrealtime stamps (100 MHz, one clock for
the whole device) patched in, each after a workgroup barrier and, where it ends a memory phase,
after s_waitcnt vmcnt(0):

- 0 start: the stream-K phase begins (persistent loop done and drained);
- 1 mac: the segment's MAC loop is done;
- 2 stored: (rs) the partial stores have completed; (owner) same as mac; (atomic) same as mac;
- 3 synced: (rs) arrived on the tile barrier and, for a reader, passed it; (owner) the owner has
  seen every peer's flag, a contributor has published; (atomic_iz) the tile's zeros are all in;
- 5 read: (rs readers) the last group's partial loads have completed, before its bias and C store;
- 4 done: the C stores (rs: the slice; owner: the whole tile on the owner) or the atomic adds
  have completed.

The stamps go after the barrier counts in the tile_sync buffer, which the harness allocates
larger. Each stamp costs a barrier and a memory drain, so absolute times are a little longer
than the real kernel's. Phases are medians over the readers (rs: with STREAMK_RS_READERS=pow2
only the first power of two of a tile's programs read), the owners (owner) or every split program
(atomic); the tail is the last done minus the first start, over all split programs. One k-step is
about 1.21 us.
"""

import argparse
import importlib.util
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import bench_ext as bench  # noqa: E402  (plugin hooks and the path to common.py)
import torch  # noqa: E402
import triton  # noqa: E402

K_STEP_US = 1.21
STAMPS = ["start", "mac", "stored", "synced", "done"]
OFFSET = 1024  # first stamp slot in tile_sync (int32), after 2 x 256 barrier words

STAMP_FN = '''

@gluon.jit
def _stamp(sync_ptr, spid, i: gl.constexpr, WAIT: gl.constexpr, BARRIER: gl.constexpr = True):
    if WAIT:
        pre: gl.constexpr = "s_waitcnt vmcnt(0)\\ns_barrier\\n"
    elif BARRIER:
        pre: gl.constexpr = "s_barrier\\n"
    else:
        pre: gl.constexpr = ""
    t = gl.inline_asm_elementwise(
        pre + "s_memrealtime $0\\ns_waitcnt lgkmcnt(0)", "=s,v,~{memory}", [_one()],
        dtype=gl.int64, is_pure=False, pack=1,
    )
    zero = gl.zeros([1], gl.int32, gl.BlockedLayout([1], [64], [gl.num_warps()], [0]))
    gl.store(sync_ptr + 1024 + spid * 8 + i + zero, t.to(gl.int32))
'''


def patch(src, version):
    """Stamps for the tile-aligned one-tile path of v20, v22 or v23 (slots in the docstring)."""
    import re

    def rep(old, new):
        nonlocal src
        assert src.count(old) == 1, old
        src = src.replace(old, new)

    rep("\n\n@gluon.jit\ndef drain_and_barrier():", STAMP_FN + "\n\n@gluon.jit\ndef drain_and_barrier():")
    rep(
        "    drain_and_barrier()\n\n    gStoreLayoutC: gl.constexpr",
        "    drain_and_barrier()\n    _stamp(sync_ptr, spid, 1, False)\n\n    gStoreLayoutC: gl.constexpr",
    )
    # Reduce-scatter: arrival, barrier passed, the slice's reads, done.
    rep("def reduce_scatter_slice(\n    p_ptr,\n", "def reduce_scatter_slice(\n    sync_ptr,\n    spid,\n    p_ptr,\n")
    src, k = re.subn(r"reduce_scatter_slice\(\n(\s+)p_ptr, ", r"reduce_scatter_slice(\n\1sync_ptr, spid, p_ptr, ", src)
    assert k == 1
    if version == 23:
        rep(
            "            drain_and_barrier()\n            is_reader = j < r_tile\n"
            "            tile_barrier(sync_ptr, spid - j, n_tile, is_reader)\n",
            "            drain_and_barrier()\n            _stamp(sync_ptr, spid, 2, False)\n"
            "            is_reader = j < r_tile\n            tile_barrier(sync_ptr, spid - j, n_tile, is_reader)\n"
            "            _stamp(sync_ptr, spid, 3, False, False)\n",
        )
        rep("                    mfmaLayout,\n                )\n",
            "                    mfmaLayout,\n                )\n            _stamp(sync_ptr, spid, 4, True)\n")
        rep("            s2 = from_pairs(s4, ROWS, mfmaLayout)\n",
            "            s2 = from_pairs(s4, ROWS, mfmaLayout)\n            _stamp(sync_ptr, spid, 5, True)\n")
        rep("            if ADD_BIAS:\n                qb = (g + ib) // 16\n",
            "            _stamp(sync_ptr, spid, 5, True)\n            if ADD_BIAS:\n                qb = (g + ib) // 16\n")
        rep(
            "        if spid < SK_PROGRAMS:\n            t, j, n, readers = streamk_chunk_of(",
            "        if spid < SK_PROGRAMS:\n            _stamp(sync_ptr, spid, 0, True)\n"
            "            t, j, n, readers = streamk_chunk_of(",
        )
    else:
        rep(
            "            drain_and_barrier()\n            tile_barrier(sync_ptr, spid - j, n_tile)\n",
            "            drain_and_barrier()\n            _stamp(sync_ptr, spid, 2, False)\n"
            "            tile_barrier(sync_ptr, spid - j, n_tile)\n            _stamp(sync_ptr, spid, 3, False, False)\n",
        )
        rep(
            "                pid_n, stride_cm, stride_cn, BLOCK_N, ADD_BIAS, RS_NB, RS_U, out_dtype,\n            )\n",
            "                pid_n, stride_cm, stride_cn, BLOCK_N, ADD_BIAS, RS_NB, RS_U, out_dtype,\n            )\n"
            "            _stamp(sync_ptr, spid, 4, True)\n",
        )
        rep("        # Block q * 16 + v: quadrant q",
            "        _stamp(sync_ptr, spid, 5, True)\n        # Block q * 16 + v: quadrant q")
        rep(
            "        if spid < SK_PROGRAMS:\n            t, j, n = streamk_chunk_of(spid, EXTRA, N_HI, N_LO)\n",
            "        if spid < SK_PROGRAMS:\n            _stamp(sync_ptr, spid, 0, True)\n"
            "            t, j, n = streamk_chunk_of(spid, EXTRA, N_HI, N_LO)\n",
        )
    # Owner (the same code in every version).
    rep(
        "        if n_peers > 0:\n            asm_fence(\"s_nop 0\")\n",
        "        if n_peers > 0:\n            asm_fence(\"s_nop 0\")\n"
        "        _stamp(sync_ptr, spid, 2, False)\n",
    )
    rep(
        "            gl.atomic_xchg(locks_ptr + spid, 1, sem=\"relaxed\", scope=\"gpu\")\n",
        "            gl.atomic_xchg(locks_ptr + spid, 1, sem=\"relaxed\", scope=\"gpu\")\n"
        "        _stamp(sync_ptr, spid, 3, False)\n",
    )
    rep(
        "            gl.atomic_xchg(locks_ptr + spid + PEER_STEP * (1 + pj), 0, sem=\"relaxed\", scope=\"gpu\")\n\n\n",
        "            gl.atomic_xchg(locks_ptr + spid + PEER_STEP * (1 + pj), 0, sem=\"relaxed\", scope=\"gpu\")\n"
        "        _stamp(sync_ptr, spid, 4, True)\n\n\n",
    )
    if version == 22:
        # Atomic: 2 before the zeroing wait (atomic_iz), 3 after it, 4 once the adds have completed.
        rep("        first = (j == 0).to(gl.float32)\n",
            "        first = (j == 0).to(gl.float32)\n        _stamp(sync_ptr, spid, 2, False)\n")
        rep("        atomic_quadrant(acc_tl, 0,", "        _stamp(sync_ptr, spid, 3, False, False)\n        atomic_quadrant(acc_tl, 0,")
        rep("ADD_BIAS, out_dtype, atomLayoutC, mfmaLayout)\n        if ZERO_IN_KERNEL:\n",
            "ADD_BIAS, out_dtype, atomLayoutC, mfmaLayout)\n        _stamp(sync_ptr, spid, 4, True)\n"
            "        if ZERO_IN_KERNEL:\n")
    return src


# Two-tile (--two-tile): a program runs up to three segments, last to first, so each stamp's slot
# depends on the segment's role. 16 slots per spid: 0 start; partial segment (the head piece)
# 1 mac, 2 after the (empty) flag wait, 3 partial stores complete, 5 end (after the flag atomic
# has completed); owner segment 9 mac, 10 peer flag seen, 13 C stored. A whole-tile segment
# (owner without a peer) writes the owner slots too, and the tail segment, processed last,
# overwrites them. A stamp directly after the flag atomic makes the results wrong, so the
# release is timed from slot 3 to slot 5.
TT_SLOTS = 16
STAMP2_FN = '''

@gluon.jit
def _stamp2(sync_ptr, spid, slot, WAIT: gl.constexpr):
    if WAIT:
        pre: gl.constexpr = "s_waitcnt vmcnt(0)\\ns_barrier\\n"
    else:
        pre: gl.constexpr = "s_barrier\\n"
    t = gl.inline_asm_elementwise(
        pre + "s_memrealtime $0\\ns_waitcnt lgkmcnt(0)", "=s,v,~{memory}", [_one()],
        dtype=gl.int64, is_pure=False, pack=1,
    )
    zero = gl.zeros([1], gl.int32, gl.BlockedLayout([1], [64], [gl.num_warps()], [0]))
    gl.store(sync_ptr + 1024 + spid * 16 + slot + zero, t.to(gl.int32))
'''


def patch_two_tile(src, relaxed):
    """Stamps for the end-to-end (two-tile) path. relaxed: the contributor's flag store as in the
    kernel (written-through partials, relaxed flag); release: the compiler's release instead,
    which adds a buffer_wbl2 (the sequence before 2026-10-08)."""
    def rep(old, new):
        nonlocal src
        assert src.count(old) == 1, old
        src = src.replace(old, new)

    role = "8 * (1 - is_partial)"
    rep("\n\n@gluon.jit\ndef drain_and_barrier():", STAMP2_FN + "\n\n@gluon.jit\ndef drain_and_barrier():")
    rep(
        "    drain_and_barrier()\n\n    gStoreLayoutC: gl.constexpr",
        f"    drain_and_barrier()\n    _stamp2(sync_ptr, spid, 1 + {role}, False)\n\n    gStoreLayoutC: gl.constexpr",
    )
    rep(
        "        if n_peers > 0:\n            asm_fence(\"s_nop 0\")\n",
        "        if n_peers > 0:\n            asm_fence(\"s_nop 0\")\n"
        f"        _stamp2(sync_ptr, spid, 2 + {role}, False)\n",
    )
    sem = "relaxed" if relaxed else "release"
    rep(
        "            drain_and_barrier()\n            gl.atomic_xchg(locks_ptr + spid, 1, sem=\"relaxed\", scope=\"gpu\")\n",
        "            drain_and_barrier()\n            _stamp2(sync_ptr, spid, 3, False)\n"
        f"            gl.atomic_xchg(locks_ptr + spid, 1, sem=\"{sem}\", scope=\"gpu\")\n",
    )
    rep(
        "            gl.atomic_xchg(locks_ptr + spid + PEER_STEP * (1 + pj), 0, sem=\"relaxed\", scope=\"gpu\")\n\n\n",
        "            gl.atomic_xchg(locks_ptr + spid + PEER_STEP * (1 + pj), 0, sem=\"relaxed\", scope=\"gpu\")\n"
        f"        _stamp2(sync_ptr, spid, 5 + {role}, True)\n\n\n",
    )
    rep(
        "    gl.amd.cdna4.async_copy.wait_group(0)\n\n    if END_TO_END and STREAMK_TILES > 0:",
        "    gl.amd.cdna4.async_copy.wait_group(0)\n"
        "    _stamp2(sync_ptr, get_logical_chiplet_mapped_pids(NUM_PROGRAMS, NUM_XCDS), 0, True)\n\n"
        "    if END_TO_END and STREAMK_TILES > 0:",
    )
    return src


def load_patched(two_tile=False, relaxed=False, version=23):
    src = open(os.path.join(os.path.dirname(HERE), bench.VERSION_MAP[version], "matmul_kernel.py")).read()
    d = tempfile.mkdtemp(prefix=f"v{version}_timing_")
    path = os.path.join(d, f"v{version}_timed_kernel.py")
    with open(path, "w") as f:
        f.write(patch_two_tile(src, relaxed) if two_tile else patch(src, version))
    spec = importlib.util.spec_from_file_location(f"v{version}_timed_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def split_counts(M, N, K, s=None, snp=256):
    """Programs per one-tile stream-K tile, per spid (0 for idle spids), as in the kernel, with snp
    programs taking part (STREAMK_NUM_PROGRAMS)."""
    s = s or (M // 256) * (N // 256) % 256
    pairs = K // 128
    n_lo, n_hi = min(snp // s, pairs), min(snp // s + 1, pairs)
    extra = snp % s if n_hi > n_lo else 0
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
        t0 = min(int(st[p, 0]) for p, (n, _) in enumerate(counts) if n >= 2)
        rel = [[(int(st[p, i]) - t0) % 2**32 * 0.01 for i in range(5)] for p in range(256)]
        last_done = 0.0
        for p, (n, j) in enumerate(counts):
            if n < 2:
                continue
            r = rel[p]
            last_done = max(last_done, r[4])
            if fixup == "owner" and j != 0:
                continue
            if fixup == "rs" and j >= readers_of(n, 23, {"STREAMK_RS_READERS": os.environ.get("STREAMK_RS_READERS", "pow2")}):
                continue
            phases["mac"].append(r[1] - r[0])
            phases["store"].append((r[2] - r[1]) if fixup == "rs" else 0.0)
            phases["sync"].append(r[3] - r[2] if fixup == "rs" else r[2] - r[1])
            phases["fixup"].append(r[4] - r[3] if fixup == "rs" else r[4] - r[2])
        phases["tail"].append(last_done)
    ms = triton.testing.do_bench(lambda: mod.matmul(a, b), quantiles=[0.5])
    ms = ms[0] if isinstance(ms, (list, tuple)) else ms
    return {k: median(v) for k, v in phases.items()}, ms * 1e3


LIMIT_MATRIX = (4096, 8192)
# E2 (v23 WORKLOG): sweep E1's cells at their best reduce-scatter n.
E2_CASES = ("1:16384:64", "8:8192:16", "16:8192:8", "32:4096:4", "64:8192:4", "12:8192:12", "24:8192:6")
E2_CONFIGS = ("20:rs", "23:rs", "23:rs+STREAMK_PARTIALS=bf16", "23:rs+STREAMK_PARTIALS=bf16_pairs", "20:owner",
              "23:owner+STREAMK_PARTIALS=bf16_pairs", "22:atomic", "22:atomic_iz")
ENV_KEYS = ("STREAMK_NUM_PROGRAMS", "STREAMK_POLICY", "STREAMK_FIXUP", "STREAMK_BALANCE_XCDS", "GEMM_TILE_LIMIT",
            "STREAMK_PARTIALS", "STREAMK_RS_READERS")


def parse_config(spec):
    """(version, fixup, {NAME: VALUE}) from VERSION:FIXUP[+NAME=VALUE...]."""
    v, rest = spec.split(":", 1)
    fixup, *kv = rest.split("+")
    return int(v), fixup, dict(x.split("=", 1) for x in kv)


def readers_of(n, version, extra):
    """Programs of an n-way split tile that read a slice: v23 with power-of-two readers reads with the
    largest power of two <= n, everything else with all n."""
    if version == 23 and extra.get("STREAMK_RS_READERS", "pow2") == "pow2":
        return 1 << (n.bit_length() - 1)
    return n


def measure_limited(mod, version, K, S, n, fixup, extra, launches):
    """One-tile phases on LIMIT_MATRIX x K with GEMM_TILE_LIMIT = 256 + S and n programs per tile
    (STREAMK_NUM_PROGRAMS = S n). Medians over the readers (rs), owners (owner) or every split program
    (atomic) and the launches after the first, plus the spread of the programs' stream-K start times:
    over all of them, and within a tile. tail is the last done minus the first start. Endings that
    round more than once (atomic, bf16 partials) are checked by their mean error, as check_tile_limit.py
    does; the others with bench_ext.py's tolerance."""
    M, N = LIMIT_MATRIX
    T = 256 + S
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    os.environ.update(STREAMK_FIXUP=fixup, STREAMK_POLICY="one_tile", STREAMK_NUM_PROGRAMS=str(S * n),
                      GEMM_TILE_LIMIT=str(T), **extra)
    a = torch.rand((M, K), device=bench.DEVICE, dtype=torch.bfloat16) - 0.5
    b = torch.rand((N, K), device=bench.DEVICE, dtype=torch.bfloat16).T - 0.5
    c = torch.empty((M, N), device=bench.DEVICE, dtype=torch.bfloat16)
    P, locks, tile_sync = mod.streamk_workspace(a.device, 256, 256, 256)
    if tile_sync.numel() < OFFSET + 256 * 8:
        bigger = torch.zeros(OFFSET + 256 * 8, device=a.device, dtype=torch.int32)
        mod._WORKSPACE[(str(a.device), 256, 256, 256)] = (P, locks, bigger)
        tile_sync = bigger
    ref32 = a.float() @ b.float()
    ref = ref32.to(torch.bfloat16)
    many_roundings = fixup.startswith("atomic") or extra.get("STREAMK_PARTIALS", "fp32") != "fp32"
    counts = split_counts(M, N, K, s=S, snp=S * n)
    tile_of, t = [], -1
    for nn, j in counts:
        t += nn > 0 and j == 0
        tile_of.append(t)
    keys = ("mac", "store", "sync", "read", "cstore", "collect", "ending", "tail", "start_spread", "tile_start_spread")
    ph = {k: [] for k in keys}
    wrong = 0
    for i in range(launches + 1):
        tile_sync[OFFSET:].zero_()
        c.fill_(float("nan"))
        mod.matmul(a, b, c)
        torch.cuda.synchronize()
        written = ~torch.isnan(c)
        if int(written.sum()) != T * 256 * 256:
            wrong += 1
        elif many_roundings:
            err = (c.float() - ref32).abs()[written].mean()
            rnd = (ref.float() - ref32).abs()[written].mean()
            wrong += bool(err > (1 + n ** 0.5) * rnd)
        else:
            wrong += not torch.allclose(torch.where(written, c, ref), ref, atol=1e-1, rtol=1e-2)
        if i == 0:
            continue
        st = tile_sync[OFFSET:OFFSET + 256 * 8].view(256, 8)[:, :6].to(torch.int64).cpu().tolist()
        active = [p for p, (nn, _) in enumerate(counts) if nn >= 2]
        t0 = min(st[p][0] for p in active)
        rel = {p: [((x - t0) % 2**32) * 0.01 for x in st[p][:6]] for p in active}
        starts = [rel[p][0] for p in active]
        ph["start_spread"].append(max(starts) - min(starts))
        by_tile = {}
        for p in active:
            by_tile.setdefault(tile_of[p], []).append(rel[p][0])
        ph["tile_start_spread"] += [max(v) - min(v) for v in by_tile.values()]
        ph["tail"].append(max(rel[p][4] for p in active))
        for p in active:
            r = rel[p]
            nn, j = counts[p]
            if fixup == "owner" and j != 0:
                continue
            if fixup == "rs" and j >= readers_of(nn, version, extra):
                continue
            ph["mac"].append(r[1] - r[0])
            ph["ending"].append(r[4] - r[1])
            if fixup == "owner":
                ph["store"].append(0.0)
                ph["sync"].append(r[2] - r[1])
                ph["collect"].append(r[4] - r[2])
                continue
            ph["store"].append(r[2] - r[1])
            ph["sync"].append(r[3] - r[2])
            ph["collect"].append(r[4] - r[3])
            if fixup == "rs":
                # A reader without blocks (more than 64 readers) has no read stamp.
                read_end = r[5] if st[p][5] else r[3]
                ph["read"].append(read_end - r[3])
                ph["cstore"].append(r[4] - read_end)
    out = {k: median(v) for k, v in ph.items()}
    out["wrong"] = wrong
    return out


def main_limited(args):
    import json

    cases = [tuple(int(x) for x in c.split(":")) for c in args.cases]
    configs = [parse_config(c) for c in args.configs]
    mods = {v: load_patched(version=v) for v in sorted({v for v, _, _ in configs})}
    print(f"One-tile phases on {LIMIT_MATRIX[0]}x{LIMIT_MATRIX[1]}xK with GEMM_TILE_LIMIT = 256 + S, n programs "
          "per tile. Medians in us over the readers (rs), the owners (owner) or all split programs (atomic) and "
          f"{args.launches} launches. publish: the partial stores; wait: the tile barrier (rs), the last peer flag "
          "(owner) or the tile's zeros (atomic_iz); reads / C store: the slice's loads, then its bias and C store "
          "(rs); collect: wait done to C stored, or the atomic adds; ending: MAC done to C stored; start spread: "
          "the last program's stream-K start minus the first's, over the tail and within a tile; tail: the last "
          "done minus the first start.\n")
    print("| K | S | n | config | MAC | publish | wait | reads | C store | collect | ending | start spread | "
          "start spread in a tile | tail | tail - MAC - ending | wrong |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for S, K, n in cases:
        for spec, (v, fixup, extra) in zip(args.configs, configs):
            r = measure_limited(mods[v], v, K, S, n, fixup, extra, args.launches)

            def f(key):
                x = r[key]
                return "-" if x != x else f"{x:.1f}"

            print(f"| {K} | {S} | {n} | {spec} | {f('mac')} | {f('store')} | {f('sync')} | {f('read')} | "
                  f"{f('cstore')} | {f('collect')} | {f('ending')} | {f('start_spread')} | {f('tile_start_spread')} | "
                  f"{f('tail')} | {r['tail'] - r['mac'] - r['ending']:.1f} | {r['wrong']} |", flush=True)
            if args.jsonl:
                with open(args.jsonl, "a") as fh:
                    fh.write(json.dumps(dict(K=K, S=S, n=n, config=spec, **r)) + "\n")


def between_modes(a, b):
    """Work before each launch, as in experiments/streamk_cache_state/modes.py."""
    big = torch.empty(256 * 2**20 // 4, device=a.device, dtype=torch.int32)

    def warm():
        a.sum()
        b.sum()

    return {
        "none": lambda: None,
        "clean_cold": lambda: big.sum(),
        "dirty_cold": lambda: big.zero_(),
        "clean_warm": lambda: (big.sum(), warm()),
        "dirty_warm": lambda: (big.zero_(), warm()),
    }


def two_tile_peers(M, N, K):
    """Per spid: does its last-processed (owner) segment have a peer? As the kernel's end-to-end
    partition with all 256 programs: the owner has a peer when its range starts inside a tile."""
    total = (M // 256) * (N // 256)
    s = total % 256 + 256
    ppt = K // 128
    L, R = s * ppt // 256, s * ppt % 256
    return [(p * L + min(p, R)) % ppt != 0 for p in range(256)]


def measure_two_tile(mod, M, N, K, mode, launches):
    """Phase medians (us) of the two-tile path after `mode`'s work before every launch."""
    os.environ["STREAMK_POLICY"] = "two_tile"
    a = torch.rand((M, K), device=bench.DEVICE, dtype=torch.bfloat16) - 0.5
    b = torch.rand((N, K), device=bench.DEVICE, dtype=torch.bfloat16).T - 0.5
    c = torch.empty((M, N), device=bench.DEVICE, dtype=torch.bfloat16)
    P, locks, tile_sync = mod.streamk_workspace(a.device, 256, 256, 256)
    if tile_sync.numel() < OFFSET + 256 * TT_SLOTS:
        bigger = torch.zeros(OFFSET + 256 * TT_SLOTS, device=a.device, dtype=torch.int32)
        mod._WORKSPACE[(str(a.device), 256, 256, 256)] = (P, locks, bigger)
        tile_sync = bigger
    ref = torch.matmul(a, b)
    between = between_modes(a, b)[mode]
    peers = two_tile_peers(M, N, K)
    ph = {k: [] for k in ("head_mac", "store", "release", "owner_mac", "owner_wait", "owner_fixup", "tail", "kernel_us")}
    wrong = 0
    for i in range(launches + 1):
        tile_sync[OFFSET:].zero_()
        between()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        mod.matmul(a, b, c)
        e.record()
        torch.cuda.synchronize()
        wrong += not torch.allclose(c, ref, atol=1e-1, rtol=1e-2)
        if i == 0:
            continue
        st = tile_sync[OFFSET:OFFSET + 256 * TT_SLOTS].view(256, TT_SLOTS).to(torch.int64).cpu().tolist()
        t0 = min(r[0] % 2**32 for r in st if r[0])

        def t(x):
            return ((x - t0) % 2**32) * 0.01

        last = 0.0
        for p, r in enumerate(st):
            last = max([last] + [t(x) for x in r if x])
            if r[1] and r[3] and r[5]:
                ph["head_mac"].append(t(r[1]) - t(r[0]))
                ph["store"].append(t(r[3]) - t(r[2]))
                ph["release"].append(t(r[5]) - t(r[3]))
            if peers[p] and r[9] and r[10] and r[13]:
                ph["owner_mac"].append(t(r[9]) - t(r[5] if r[5] else r[0]))
                ph["owner_wait"].append(t(r[10]) - t(r[9]))
                ph["owner_fixup"].append(t(r[13]) - t(r[10]))
        ph["tail"].append(last)
        ph["kernel_us"].append(s.elapsed_time(e) * 1e3)
    out = {k: median(v) for k, v in ph.items()}
    rel = sorted(ph["release"])
    out["release_p90"] = rel[len(rel) * 9 // 10] if rel else float("nan")
    out["wrong"] = wrong
    return out


def main_two_tile(args):
    import json

    mods = {flag: load_patched(two_tile=True, relaxed=flag == "relaxed") for flag in args.flags}
    print("Two-tile phases, medians in us over programs and launches. head MAC: the head piece; store: "
          "partial stores completing; release: the flag atomic; owner: the tail segment of a program "
          "whose range starts inside a tile.\n")
    print("| shape | flag | mode | head MAC | store | release (p90) | owner MAC | owner wait | owner fixup | tail "
          "| kernel (events) | wrong |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in args.shapes:
        M, N, K = (int(x) for x in s.split("x"))
        for flag, mod in mods.items():
            for mode in args.modes:
                r = measure_two_tile(mod, M, N, K, mode, args.launches)
                print(f"| {s} | {flag} | {mode} | {r['head_mac']:.1f} | {r['store']:.1f} | {r['release']:.2f} "
                      f"({r['release_p90']:.2f}) | {r['owner_mac']:.1f} | {r['owner_wait']:.1f} | "
                      f"{r['owner_fixup']:.1f} | {r['tail']:.1f} | {r['kernel_us']:.1f} | {r['wrong']} |", flush=True)
                if args.jsonl:
                    with open(args.jsonl, "a") as f:
                        f.write(json.dumps(dict(shape=s, flag=flag, mode=mode, **r)) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--shapes", nargs="*",
        default=["3328x5120x8192", "4352x4096x4096", "4352x4096x8192", "4352x4352x8192", "4096x5120x8192",
                 "4096x6144x8192"],
    )
    parser.add_argument("--launches", type=int, default=6)
    parser.add_argument("--two-tile", action="store_true", help="time the two-tile path's publish instead")
    parser.add_argument("--modes", nargs="+", default=["clean_cold", "dirty_cold"],
                        help="--two-tile: work before each launch (none, clean_cold, dirty_cold, clean_warm, dirty_warm)")
    parser.add_argument("--flags", nargs="+", default=["release", "relaxed"], choices=["release", "relaxed"],
                        help="--two-tile: the contributor's flag store (release is the pre-2026-10-08 control)")
    parser.add_argument("--jsonl", help="--two-tile, --tile-limit: append one JSON line per row to this file")
    parser.add_argument("--tile-limit", action="store_true",
                        help="the one-tile split on one matrix, S leftover tiles by GEMM_TILE_LIMIT (sweep C's matrix)")
    parser.add_argument("--cases", nargs="+", default=list(E2_CASES), help="--tile-limit: S:K:n cells")
    parser.add_argument("--configs", nargs="+", default=list(E2_CONFIGS),
                        help="--tile-limit: VERSION:FIXUP[+NAME=VALUE...] configurations")
    args = parser.parse_args()
    if args.two_tile:
        main_two_tile(args)
        return
    if args.tile_limit:
        main_limited(args)
        return
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
