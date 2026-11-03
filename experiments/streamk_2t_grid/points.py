"""Point lists for the reduced-grid two-tile sweep (CPU only).

    python experiments/streamk_2t_grid/points.py      # writes results/pilot.txt, results/matrix.txt

A point is experiments/streamk_v31_rule's ``KIND:BMxBNxBK:MxNxK``; ``2tg{G}`` is two_tile
on G programs (G < T < 2 G, a multiple of 8), ``def`` the v31 default plan. Every shape
also gets dp, ``2t`` (G = 256) and every valid one-tile owner and reduce-scatter split.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "streamk_v31_rule"))
import grid  # noqa: E402

TILE = "256x256x64"
# (S label, M, N): T = 256 + S with 256x256 tiles, plus two sub-wave shapes.
SHAPES = [
    (16, 4096, 4352),
    (32, 4096, 4608),
    (44, 3840, 5120),
    (64, 4096, 5120),
    (80, 4096, 5376),
    (96, 4096, 5632),
    (128, 4096, 6144),
    ("sub200", 2560, 5120),
    ("sub160", 2560, 4096),
]
KS = (1024, 2048, 4096, 8192)
LONG_K = {44: (16384,), 96: (16384,)}


def grids(T):
    """Every multiple of 8 with T / 2 < G < T, G <= 256 (G = 256 is ``2t``)."""
    return [G for G in range(8, min(T - 1, grid.P - 1) + 1, 8) if 2 * G > T]


def kinds(M, N, K):
    T = grid.shape_info(TILE, M, N, K)["T"]
    out = ["dp", "def", "2t"] + [f"ow{n}" for n in grid.OWNER_N] + [f"rs{n}" for n in grid.RS_N]
    out += [f"2tg{G}" for G in grids(T)]
    return [k for k in out if grid.valid(k, TILE, M, N, K)]


def exact(T, pairs, G):
    """r = T / G in lowest terms a / b, and whether every cut falls on a multiple of
    pairs / b (T pairs divisible by G and pairs by b)."""
    b = G // math.gcd(T, G)
    return b, (T * pairs) % G == 0 and pairs % b == 0


def points(shapes, ks):
    out = []
    for s, M, N in shapes:
        for K in ks + LONG_K.get(s, ()):
            out += [grid.point_name(k, TILE, M, N, K) for k in kinds(M, N, K)]
    return out


def main():
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    pilot = points([sh for sh in SHAPES if sh[0] == 44], (2048, 8192))
    pilot = [p for p in pilot if not p.endswith("x16384")]
    matrix = points(SHAPES, KS)
    for name, pts in (("pilot", pilot), ("matrix", matrix)):
        with open(os.path.join(HERE, "results", f"{name}.txt"), "w") as f:
            f.write("\n".join(pts) + "\n")
        print(f"{name}: {len(pts)} points")


if __name__ == "__main__":
    main()
