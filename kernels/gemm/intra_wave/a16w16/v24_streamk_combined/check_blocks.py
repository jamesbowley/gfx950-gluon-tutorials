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
"""Pure-Python check of v24's placements (no GPU), after v23's address-map checks.

    python v24_streamk_combined/check_blocks.py

1-5. v23's checks (v23_streamk_rs_band/check_blocks.py): the partial and slice address maps, the
   reduce-scatter coverage, group shapes and readers, which v24 does not change.
6. Placement. For every leftover-tile count S, program count S <= P <= 256 and K, under tile-major
   and chunk-major placement, replaying the kernel's decode (streamk_chunk_of /
   streamk_chunk_major_of, the role rotation and streamk_slot) for every stream-K id:
   - every program gets a unique (tile, chunk), every tile its chunks 0 .. n - 1 exactly once,
     with n as streamk_chunk_of gives it;
   - every tile has exactly one role 0 (owner, first slice, bias), and its roles are 0 .. n - 1;
   - the slots of tile t are exactly streamk_slot(t, 0) .. + n - 1, every slot is used once, and
     the slots fit the workspace (fewer than 256);
   - under tile-major placement slot = stream-K id and role = chunk (v23's code);
   - under chunk-major placement role 0 is chunk t % n.
   K enters only through the cap of one program per pair of k-steps, so K = 512, 2048 and 8192
   (4, 16 and 64 pairs) cover the capped and uncapped cases.
7. XCD-local split (STREAMK_EXCHANGE=l2): for every S and n the host can pass, the tiles of
   xcd_local_chunk_of stay on one XCD, every tile gets xcd_local_split programs, and tile_groups
   (the host's straddle check) agrees.
"""

import importlib.util
import itertools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("v23_check_blocks",
                                              os.path.join(HERE, "..", "v23_streamk_rs_band", "check_blocks.py"))
v23cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v23cb)

NUM_PROGRAMS, NUM_XCDS = 256, 8
PER = NUM_PROGRAMS // NUM_XCDS
KS = (512, 1024, 2048, 4096, 8192, 16384, 32768)


def split_consts(S, snp, pairs):
    """The kernel's N_LO, N_HI, EXTRA and SK_PROGRAMS for the tile-aligned split."""
    n_lo = min(snp // S, pairs)
    n_hi = min(snp // S + 1, pairs)
    extra = snp % S if n_hi > n_lo else 0
    return n_lo, n_hi, extra, extra * n_hi + (S - extra) * n_lo


def chunk_of(cid, extra, n_hi, n_lo):
    head = extra * n_hi
    if cid < head:
        return cid // n_hi, cid % n_hi, n_hi
    return extra + (cid - head) // n_lo, (cid - head) % n_lo, n_lo


def chunk_major_of(cid, S, extra, n_hi, n_lo):
    if cid >= n_lo * S:
        t, j = cid - n_lo * S, n_lo
    else:
        t, j = cid % S, cid // S
    return t, j, (n_hi if t < extra else n_lo)


def slot_of(t, i, extra, n_hi, n_lo):
    b = t * n_hi if t < extra else extra * n_hi + (t - extra) * n_lo
    return b + i


def decode(cid, S, extra, n_hi, n_lo, chunk_major):
    """(t, j, n, role, slot) of stream-K id cid, as the kernel's tile-aligned dispatch computes them."""
    if chunk_major:
        t, j, n = chunk_major_of(cid, S, extra, n_hi, n_lo)
        role = (j + n - t % n) % n
        return t, j, n, role, slot_of(t, role, extra, n_hi, n_lo)
    t, j, n = chunk_of(cid, extra, n_hi, n_lo)
    return t, j, n, j, cid


def check_placement():
    for S, k in itertools.product(range(1, 257), (512, 2048, 8192)):
        pairs = k // 128
        for snp in range(S, 257):
            n_lo, n_hi, extra, skp = split_consts(S, snp, pairs)
            want_n = [n_hi if t < extra else n_lo for t in range(S)]
            for cm in (False, True):
                chunks = [set() for _ in range(S)]
                roles = [set() for _ in range(S)]
                slots = {}
                for cid in range(skp):
                    t, j, n, role, slot = decode(cid, S, extra, n_hi, n_lo, cm)
                    where = f"S={S} P={snp} K={k} {'chunk' if cm else 'tile'} id {cid}"
                    if not 0 <= t < S or n != want_n[t]:
                        return f"{where}: tile {t}, n {n}"
                    if j in chunks[t] or role in roles[t] or slot in slots:
                        return f"{where}: repeated chunk, role or slot"
                    if not cm and (slot != cid or role != j):
                        return f"{where}: tile-major slot or role is not v23's"
                    if cm and role == 0 and j != t % n:
                        return f"{where}: role 0 is chunk {j}, not {t % n}"
                    chunks[t].add(j)
                    roles[t].add(role)
                    slots[slot] = (t, role)
                for t in range(S):
                    n = want_n[t]
                    if chunks[t] != set(range(n)) or roles[t] != set(range(n)):
                        return f"S={S} P={snp} K={k}: tile {t} chunks or roles are not 0..{n - 1}"
                    b = slot_of(t, 0, extra, n_hi, n_lo)
                    if any(slots.get(b + i) != (t, i) for i in range(n)):
                        return f"S={S} P={snp} K={k}: tile {t} slots are not {b}..{b + n - 1}"
                if max(slots) >= NUM_PROGRAMS:
                    return f"S={S} P={snp} K={k}: slot {max(slots)} outside the workspace"
    return None


def check_xcd_local():
    sys.path.insert(0, os.path.dirname(HERE))
    import bench_ext as bench  # noqa: F401  (the path to common.py, which the kernel module imports)
    from v24_streamk_combined import matmul_kernel as mk

    for S, k in itertools.product(range(1, 257), KS):
        pairs = k // 128
        for n_cap in range(1, NUM_PROGRAMS // S + 1):
            split = mk.xcd_local_split(S, n_cap, pairs, NUM_PROGRAMS, NUM_XCDS)
            if -(-S // NUM_XCDS) * split > PER:
                return f"S={S} n={n_cap} K={k}: {split} programs per tile do not fit an XCD"
            groups = mk.tile_groups(S, S * split, pairs, NUM_PROGRAMS, NUM_XCDS, True, True, split)
            if any(len(g) != split or len({s // PER for s in g}) != 1 for g in groups):
                return f"S={S} n={n_cap} K={k}: a tile has the wrong programs or spans XCDs"
            if sorted(s for g in groups for s in g) != sorted(set(s for g in groups for s in g)):
                return f"S={S} n={n_cap} K={k}: a program is on two tiles"
    return None


def main():
    failures = 0
    checks = (("decode", v23cb.check_decode), ("coverage", v23cb.check_coverage),
              ("group shape", v23cb.check_group_shape), ("pair map", v23cb.check_pair_map),
              ("readers", v23cb.check_readers), ("placement", check_placement), ("xcd-local", check_xcd_local))
    for name, check in checks:
        err = check()
        print(f"{name}: {'ok' if err is None else 'FAIL: ' + err}", flush=True)
        failures += err is not None
    print("\nPASS" if failures == 0 else "\nFAIL")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
