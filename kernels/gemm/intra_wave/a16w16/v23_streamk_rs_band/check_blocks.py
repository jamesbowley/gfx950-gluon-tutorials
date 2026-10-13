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
"""Pure-Python check of v23's reduce-scatter address maps (no GPU); v20's checks plus the
bf16 pair map, the readers and the band groups.

    python v23_streamk_rs_band/check_blocks.py

1. Decode. The writer stores element (row, col) of quadrant q at
   q * 16384 + lane_contiguous_offsets(row, col). reduce_scatter_slice reads block
   blk = address // 1024 and decodes it back to a tile row and column. Every one of the
   65536 elements of a slot must come back to where it was written, and every address must
   be hit once.
2. Coverage. For every split count n from 2 to 128, the blocks [64 j / n, 64 (j + 1) / n)
   of programs j = 0 .. n - 1 must cover the 64 blocks of a tile exactly once.
3. Group shape. rs_split must give a power-of-two NB, and NB * U loads in flight, for every
   leftover-tile count, program count and K the host can pass; with the band store, every
   reader's slice must be whole groups of NB aligned blocks.
4. Pair map. The writer stores element (row, col) of quadrant q at
   q * 16384 + pair_offsets(row, col) (to_pairs, then pair_offsets). The slice's per-block form of
   that map (pair_in_block plus pair_block_offset) must agree with it for all 65536 elements, cover
   the slot exactly once, and put each lane's 4 consecutive columns at 4 consecutive addresses
   (8 bytes) in its block reads; the pair-layout band read of aligned blocks g .. g + NB - 1 must
   cover exactly quadrant rows (g % 16) * 8 .. + 8 NB.
5. Readers. With R readers (the largest power of two <= n), the blocks [64 j / R, 64 (j + 1) / R)
   cover a tile exactly once for every n from 2 to 128.
"""

import itertools
import sys


def lane_contiguous(r, c):
    """lane_contiguous_offsets in matmul_kernel.py, for one element of a 128x128 quadrant."""
    f = (r % 16) * 4 + (r // 16 % 2) * 512 + (r // 32 % 2) * 4096 + (r // 64) * 8192
    g = c + (c // 4 % 4) * 60 + (c // 16 % 2) * 240 + (c // 32 % 2) * 992 + (c // 64) * 1984
    return f + g


def in_block(r, c):
    """reduce_scatter_slice's in_blk, for one element of a 32x32 block."""
    return (r % 16) * 4 + (r // 16) * 512 + c + (c // 4 % 4) * 60 + (c // 16) * 240


def block_origin(blk):
    """reduce_scatter_slice's (row, col) of block blk's first element in the tile."""
    q, v = blk // 16, blk % 16
    row = (q % 2) * 128 + (v // 4 % 2) * 32 + (v // 8) * 64
    col = (q // 2) * 128 + (v % 2) * 32 + (v // 2 % 2) * 64
    return row, col


def pow2_floor(x):
    return 1 << (x.bit_length() - 1)


def rs_split(streamk_tiles, num_programs, pairs_per_tile, pow2_readers, loads_in_flight=16):
    """matmul_kernel.rs_split, without importing triton."""
    s = max(streamk_tiles, 1)
    n_lo = min(num_programs // s, pairs_per_tile)
    n_hi = min(num_programs // s + 1, pairs_per_tile)
    extra = num_programs % s if n_hi > n_lo else 0
    readers = pow2_floor if pow2_readers else (lambda n: n)
    r_hi, r_lo = readers(max(n_hi, 1)), readers(max(n_lo, 1))
    r_max = r_hi if extra > 0 else r_lo
    band = all(r & (r - 1) == 0 for r in ((r_hi, r_lo) if extra > 0 else (r_lo,)))
    if band:
        nb = max(1, min(64 // r_max, loads_in_flight))
    else:
        nb = min(1 << (-(-64 // max(r_lo, 2)) - 1).bit_length(), loads_in_flight)
    return dict(n_hi=n_hi, n_lo=n_lo, extra=extra, r_hi=r_hi, r_lo=r_lo, band=band, nb=nb,
                u=max(loads_in_flight // nb, 1))


def pair_offset(r, c):
    """pair_offsets in matmul_kernel.py for element (r, c) of a quadrant: to_pairs puts it at
    (d0, d1, d2, e) = (r, c // 64, c // 4 % 8, 4 (c // 32 % 2) + c % 4)."""
    d1, d2, e = c // 64, c // 4 % 8, 4 * (c // 32 % 2) + c % 4
    vp = d1 + 2 * (r // 32 % 2) + 4 * (r // 64)
    w = d2 // 4 + 2 * (r // 16 % 2)
    lane = r % 16 + 16 * (d2 % 4)
    return ((vp * 4 + w) * 64 + lane) * 8 + e


def pair_in_block(r, c):
    """reduce_scatter_slice's in_blk for bf16 pairs, for one element of a 32x32 block."""
    return (r % 16) * 8 + (r // 16) * 1024 + c + (c // 4 % 4) * 124 + (c // 16) * 496


def pair_block_offset(blk):
    """reduce_scatter_slice's p_off - in_blk for bf16 pairs."""
    q, v = blk // 16, blk % 16
    return q * 16384 + (v % 2) * 4 + (v // 2 % 2) * 2048 + (v // 4 % 2) * 4096 + (v // 8) * 8192


def check_decode():
    decode = {in_block(r, c): (r, c) for r, c in itertools.product(range(32), range(32))}
    if sorted(decode) != list(range(1024)):
        return "in_blk is not a permutation of 0..1023"
    seen = set()
    for q, r, c in itertools.product(range(4), range(128), range(128)):
        addr = q * 16384 + lane_contiguous(r, c)
        seen.add(addr)
        row0, col0 = block_origin(addr // 1024)
        dr, dc = decode[addr % 1024]
        want = ((q % 2) * 128 + r, (q // 2) * 128 + c)
        if (row0 + dr, col0 + dc) != want:
            return f"q={q} r={r} c={c}: written at {addr}, read back as {(row0 + dr, col0 + dc)}"
    if seen != set(range(65536)):
        return "the writer does not cover the slot exactly once"
    return None


def check_coverage():
    for n in range(2, 129):
        blocks = [b for j in range(n) for b in range((64 * j) // n, (64 * (j + 1)) // n)]
        if blocks != list(range(64)):
            return f"n={n}: blocks are not covered exactly once"
    return None


def check_group_shape():
    for s, snp, k, pow2 in itertools.product(range(1, 257), range(1, 257), (512, 1024, 2048, 4096, 8192, 16384, 32768),
                                             (True, False)):
        if s > snp:
            continue
        d = rs_split(s, snp, k // 128, pow2)
        nb, u = d["nb"], d["u"]
        if nb & (nb - 1) or not 1 <= nb <= 16 or nb * u != 16:
            return f"S={s} programs={snp} K={k}: NB={nb} U={u}"
        if pow2 and not d["band"]:
            return f"S={s} programs={snp} K={k}: power-of-two readers without the band"
        if d["band"]:
            for r in {d["r_lo"], d["r_hi"] if d["extra"] else d["r_lo"]}:
                for j in range(r):
                    lo, hi = 64 * j // r, 64 * (j + 1) // r
                    if lo % nb or (hi - lo) % nb:
                        return f"S={s} programs={snp} K={k}: R={r} reader {j} blocks {lo}..{hi} not whole groups of {nb}"
    return None


def check_pair_map():
    seen = set()
    for q, r, c in itertools.product(range(4), range(128), range(128)):
        addr = q * 16384 + pair_offset(r, c)
        seen.add(addr)
        v = (c // 32 % 2) + 2 * (c // 64) + 4 * (r // 32 % 2) + 8 * (r // 64)
        got = pair_block_offset(q * 16 + v) + pair_in_block(r % 32, c % 32)
        if got != addr:
            return f"q={q} r={r} c={c}: written at {addr}, read at {got}"
    if seen != set(range(65536)):
        return "the pair map does not cover the slot exactly once"
    for r, c0 in itertools.product(range(32), range(0, 32, 4)):
        a = [pair_in_block(r, c0 + i) for i in range(4)]
        if a != list(range(a[0], a[0] + 4)) or a[0] % 4:
            return f"block row {r}, columns {c0}..{c0 + 3}: not 4 aligned consecutive elements"
    for nb in (4, 8, 16):
        for g in range(0, 64, nb):
            rows = {(g % 16) * 8 + x for x in range(8 * nb)}
            want = set()
            for blk in range(g, g + nb):
                v = blk % 16
                want |= {(v // 4 % 2) * 32 + (v // 8) * 64 + x for x in range(32)}
            if rows != want:
                return f"NB={nb} g={g}: band rows {sorted(rows)[:3]}.. are not the group's"
    return None


def check_readers():
    for n in range(2, 129):
        r = pow2_floor(n)
        blocks = [b for j in range(r) for b in range((64 * j) // r, (64 * (j + 1)) // r)]
        if blocks != list(range(64)):
            return f"n={n}, R={r}: blocks are not covered exactly once"
    return None


def main():
    failures = 0
    for name, check in (("decode", check_decode), ("coverage", check_coverage), ("group shape", check_group_shape),
                        ("pair map", check_pair_map), ("readers", check_readers)):
        err = check()
        print(f"{name}: {'ok' if err is None else 'FAIL: ' + err}")
        failures += err is not None
    print("\nPASS" if failures == 0 else "\nFAIL")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
