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
"""Predicted MFMA ceilings for small tiles (README §3, plus MFMA padding and the LDS read rate).

    python v9_tile/model_small.py            # compute-bound table, cold and warm
    python v9_tile/model_small.py mb         # memory-bound bandwidth table

A tile is BM x BN x BK on W waves, laid out [1, W] along N, with the MFMA padded to 16 rows
(and 16 columns). `tiles_cu` tiles share a CU: 4 / W when pinned to one wave per SIMD. Every
ceiling is a fraction of the time the MFMAs are busy; `useful` multiplies by the padding.

- pad:    BM BN / (Mp Np), Mp = max(16, BM), Np = max(16, BN).
- proc:   the TCP processes 64 B per clock: 64 C_cyc (W / 4) / B.
- window: 32 KB in flight per CU over the round trip L: Q C (W / 4) / (B L).
- stages: the CU's tiles hold `stages` k-steps each in flight, which must cover L plus the issue
          time of their requests: stages tiles_cu (W / 4) C / (L + T_issue). Pinned, tiles_cu
          (W / 4) = 1: stages C / (L + T_issue), README §3's ceiling.
- lds:    one ds_read_b128 (1 KB) per wave per 16 cycles against one 16x16x32 MFMA per 16
          cycles: a wave tile wm x wn reads (wm + wn) BK 2 bytes for wm wn BK / 8192 MFMAs,
          so eff <= wm wn / (16 (wm + wn)).

B counts the bytes of real rows only (the padded rows are masked lanes); with masked_cost the
TCP is charged for the padded rows too.
"""

import sys

Q = 32768
L_COLD, L_WARM = 550.0, 385.0
CLOCK = 2.1
HBM_TBS = 6.3
LDS_CU = 160 * 1024


def ceilings(BM, BN, BK, W, tiles_cu=None, L=L_COLD, clock=CLOCK, stages=2, masked_cost=False):
    Mp, Np = max(16, BM), max(16, BN)
    tiles_cu = 4 // W if tiles_cu is None else tiles_cu
    B = (BM + BN) * BK * 2
    B_tcp = (Mp + Np) * BK * 2 if masked_cost else B
    cyc = 2 * Mp * Np * BK / (1024 * W)  # MFMA cycles per k-step: W SIMDs share the tile
    C = cyc / clock
    t_issue = tiles_cu * B_tcp / 1024 * 16 / clock
    wm, wn = Mp, Np // W
    # Per CU, a tile's k-step is cyc x W SIMD-cycles of MFMA for B bytes, whatever the number of
    # tiles sharing the CU: the byte limits (processing, window) do not depend on tiles_cu. Only the
    # bytes in flight do: each tile holds `stages` k-steps.
    share = W / 4
    r = {
        "pad": BM * BN / (Mp * Np),
        "proc": min(1.0, 64 * cyc * share / B_tcp),
        "window": min(1.0, Q * C * share / (B_tcp * L)),
        "stages": min(1.0, stages * tiles_cu * share * C / (L + t_issue)),
        "lds": min(1.0, wm * wn / (16 * (wm + wn))),
    }
    r["busy"] = min(r["proc"], r["window"], r["stages"], r["lds"])
    r["binds"] = min(("proc", "window", "stages", "lds"), key=lambda k: r[k])
    r["useful"] = r["busy"] * r["pad"]
    r.update({"B_KB": B / 1024, "C_ns": C, "inflight_KB": stages * tiles_cu * B / 1024,
              "lds_KB": stages * ((Mp + Np) * BK * 2) / 1024, "tiles_cu": tiles_cu})
    return r


def two_stage_feasible(BM, BN, BK, W, tiles_cu=None, L=L_COLD, clock=CLOCK):
    """Two stages are enough when they do not bind before the window does."""
    r = ceilings(BM, BN, BK, W, tiles_cu, L, clock)
    return r["stages"] >= r["window"]


def hbm_bw(M, N, K, BM, BN, BK, W, tiles_cu=None, L=L_COLD, stages=2, cus=256, hbm=HBM_TBS):
    """Memory-bound throughput in TB/s: min(HBM, busy CUs x bytes in flight per CU / L). The
    dispatcher spreads workgroups over the CUs first, so a CU holds ceil(tiles / CUs) of them, at
    most tiles_cu."""
    tiles = -(-M // BM) * -(-N // BN)
    tiles_cu = 4 // W if tiles_cu is None else tiles_cu
    busy = min(cus, tiles)
    per_cu = min(tiles_cu, -(-tiles // busy))
    inflight = min(Q, stages * per_cu * (BM + BN) * BK * 2)
    return min(hbm, busy * inflight / L / 1e3), busy


CB_TILES = [(64, 64, 64, 4), (64, 64, 128, 4), (32, 256, 64, 4), (32, 128, 64, 4), (32, 128, 128, 4),
            (32, 64, 128, 1), (32, 32, 128, 1), (16, 256, 64, 4), (16, 256, 128, 4), (16, 128, 128, 2),
            (16, 64, 128, 1), (16, 64, 256, 1), (16, 16, 128, 1), (16, 16, 256, 1), (8, 256, 128, 4),
            (4, 256, 128, 4), (2, 256, 128, 4), (1, 256, 128, 4), (1, 64, 256, 1)]


def cb_table(L, clock=CLOCK):
    print(f"\nL = {L:.0f} ns, clock {clock} GHz, one wave per SIMD, 2 stages\n")
    print("| tile | W | tiles/CU | B KB | C ns | in flight KB | pad | proc | window | 2 stages | lds | busy | "
          "useful | binds | 2-stage ok |")
    print("|" + "---|" * 15)
    for BM, BN, BK, W in CB_TILES:
        r = ceilings(BM, BN, BK, W, L=L, clock=clock)
        ok = "yes" if r["stages"] >= r["window"] else "no"
        print(f"| {BM}x{BN}x{BK} | {W} | {r['tiles_cu']} | {r['B_KB']:.1f} | {r['C_ns']:.0f} | {r['inflight_KB']:.0f} | "
              f"{r['pad']:.0%} | {r['proc']:.0%} | {r['window']:.1%} | {r['stages']:.1%} | {r['lds']:.0%} | "
              f"{r['busy']:.1%} | {r['useful']:.1%} | {r['binds']} | {ok} |")


def mb_table(L=L_COLD):
    print(f"\nMemory-bound, L = {L:.0f} ns, pinned, 2 stages: predicted TB/s (busy CUs)\n")
    print("| M x N x K | BN=16 | 32 | 64 | 128 | 256 |")
    print("|---|---|---|---|---|---|")
    for M in (1, 16, 32):
        for N in (4096, 8192, 16384):
            cells = []
            for BN in (16, 32, 64, 128, 256):
                W = max(1, min(4, BN // 64))
                bw, busy = hbm_bw(M, N, 8192, M, BN, 128, W, L=L)
                cells.append(f"{bw:.1f} ({busy})")
            print(f"| {M}x{N}x8192 | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    if sys.argv[1:] == ["mb"]:
        mb_table()
    else:
        cb_table(L_COLD)
        cb_table(L_WARM)
