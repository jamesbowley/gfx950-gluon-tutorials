"""Correctness of two_tile on a reduced grid (reduced_grid.py).

    source kernels/gemm/intra_wave/a16w16/v31_streamk_aiter/env.sh
    python $REPO/experiments/streamk_2t_grid/check_grid.py --tile 256x256x64
    python $REPO/experiments/streamk_2t_grid/check_grid.py --tile 256x256x64 --graph
    python $REPO/experiments/streamk_2t_grid/check_grid.py --tile 256x256x64 --stale-lock

Every case checks C against an fp32 reference, the sentinel window around C, and that
the grid's locks and counts are back at zero. --stale-lock presets the locks to 1
over NaN-poisoned persistent partials: on the full launch it only reports (reversed
order stores every partial before its owner reads it), and with the last 8 programs
dropped it is the negative control and must give a wrong C.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A16 = os.path.abspath(os.path.join(HERE, "..", "..", "kernels", "gemm", "intra_wave", "a16w16"))
sys.path.insert(0, A16)
sys.path.insert(0, HERE)
import torch  # noqa: E402
import reduced_grid  # noqa: E402
from v31_streamk_aiter import matmul_kernel as mk  # noqa: E402

SENTINEL = 1234.0
NUM_XCDS = 8


def config_for(bm, bn, bk):
    return dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE_M=4, STREAMK_POLICY="two_tile",
                STREAMK_FIXUP="owner", SPLIT_N=None, STREAMK_EXCHANGE="wt")


def cases(bm, bn, bk):
    """(label, M, N, K, G, bias): T = 300 and T = 200 over exact and inexact grids,
    ragged M / N / K, bias."""
    k = 32 * bk
    return [
        ("t300_g256", 15 * bm, 20 * bn, k, 256, False),
        ("t300_g240", 15 * bm, 20 * bn, k, 240, False),
        ("t300_g200", 15 * bm, 20 * bn, k, 200, True),
        ("t300_g152", 15 * bm, 20 * bn, k, 152, False),
        ("t300_g224", 15 * bm, 20 * bn, k + 2 * bk, 224, False),
        ("t200_g136", 10 * bm, 20 * bn, k, 136, False),
        ("t200_g104", 10 * bm, 20 * bn, k, 104, True),
        ("t200_g192", 10 * bm, 20 * bn, k, 192, False),
        ("ragged", 15 * bm - bm // 8, 20 * bn - bn // 8, k - 16, 200, True),
        ("ragged_k", 15 * bm, 20 * bn, k - bk - 16, 184, False),
        ("short_k", 15 * bm, 20 * bn, 4 * bk, 200, False),
    ]


def make_case(M, N, K, bm, bn, with_bias):
    torch.manual_seed(17)
    a = torch.rand((M, K), device="cuda", dtype=torch.bfloat16) - 0.5
    w = torch.rand((N, K), device="cuda", dtype=torch.bfloat16) - 0.5
    bias = torch.rand((N,), device="cuda", dtype=torch.bfloat16) - 0.5 if with_bias else None
    backing = torch.full((M + bm, N + bn), SENTINEL, device="cuda", dtype=torch.bfloat16)
    ref = a.float() @ w.float().T
    if bias is not None:
        ref += bias.float()
    return a, w, bias, backing, ref.to(torch.bfloat16)


def flags(G):
    return mk._flags(torch.device("cuda", torch.cuda.current_device()),
                     torch.cuda.current_stream(), G, NUM_XCDS)


def rearmed(G):
    torch.cuda.synchronize()
    f = flags(G)
    return not (torch.count_nonzero(f[: 2 * G : 2]) or torch.count_nonzero(f[2 * G:]))


def run(a, w, bias, c, config, G, persistent):
    reduced_grid.gemm(a, w, bias, y=c, config=config, G=G, persistent_partials=persistent)


def verdict(c, ref, backing, M, N):
    ok = torch.allclose(c, ref, atol=0.1, rtol=0.01)
    guard = bool(torch.all(backing[M:, :] == SENTINEL) and torch.all(backing[:M, N:] == SENTINEL))
    err = torch.max(torch.abs(c.float() - ref.float())).item()
    return ok, guard, err


def check_cases(tile, launches, persistent):
    bm, bn, bk = tile
    config = config_for(*tile)
    for label, M, N, K, G, with_bias in cases(*tile):
        a, w, bias, backing, ref = make_case(M, N, K, bm, bn, with_bias)
        c = backing[:M, :N]
        for _ in range(launches):
            backing.fill_(SENTINEL)
            if persistent:
                mk._persistent_partials(torch.device("cuda", torch.cuda.current_device()),
                                        torch.cuda.current_stream(), G).fill_(float("nan"))
            run(a, w, bias, c, config, G, persistent)
        torch.cuda.synchronize()
        ok, guard, err = verdict(c, ref, backing, M, N)
        re = rearmed(G)
        T = mk.triton.cdiv(M, bm) * mk.triton.cdiv(N, bn)
        print(f"{label:10s} {M}x{N}x{K} T={T} G={G} r={T / G:.3f} bias={with_bias} "
              f"error={err:.5f} correct={ok} guard={guard} rearmed={re}", flush=True)
        if not (ok and guard and re):
            raise AssertionError(f"{label} failed")


def check_rejects(tile):
    bm, bn, bk = tile
    config = config_for(*tile)
    a = torch.zeros((15 * bm, 32 * bk), device="cuda", dtype=torch.bfloat16)
    w = torch.zeros((20 * bn, 32 * bk), device="cuda", dtype=torch.bfloat16)
    bad = [(config, 144, "T >= 2G"), (config, 204, "not a multiple of 8"),
           (config, 264, "above the CUs"), (dict(config, STREAMK_POLICY="dp"), 200, "not two_tile")]
    for cfg, G, why in bad:
        try:
            reduced_grid.gemm(a, w, config=cfg, G=G)
        except ValueError as e:
            print(f"rejected G={G} ({why}): {e}")
        else:
            raise AssertionError(f"G={G} ({why}) was accepted")


def graph_replay(tile, replays=10):
    """Two launches on different grids captured in one graph, replayed over changing
    inputs with NaN-poisoned persistent partials."""
    bm, bn, bk = tile
    config = config_for(*tile)
    sel = [c for c in cases(*tile) if c[0] in ("t300_g200", "ragged")]
    data = [(G, make_case(M, N, K, bm, bn, b), M, N) for _, M, N, K, G, b in sel]

    def launch_all():
        for G, (a, w, bias, backing, _), M, N in data:
            run(a, w, bias, backing[:M, :N], config, G, True)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        launch_all()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            launch_all()
        torch.cuda.synchronize()
        dev = torch.device("cuda", torch.cuda.current_device())
        for i in range(replays):
            for G, (a, w, bias, backing, _), M, N in data:
                a.uniform_(-0.5, 0.5)
                backing.fill_(SENTINEL)
                mk._persistent_partials(dev, s, G).fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            for G, (a, w, bias, backing, _), M, N in data:
                ref = a.float() @ w.float().T
                if bias is not None:
                    ref += bias.float()
                ok, guard, err = verdict(backing[:M, :N], ref.to(torch.bfloat16), backing, M, N)
                if not (ok and guard and rearmed(G)):
                    raise AssertionError(f"replay {i} G={G} failed: error={err}")
    print(f"graph replay passed: {replays} replays of {len(data)} launches")


class DropPrograms:
    """Stands in for the kernel: launches the grid minus its last `drop` programs."""

    def __init__(self, kernel, drop):
        self.kernel, self.drop = kernel, drop

    def __getitem__(self, grid_):
        return self.kernel[(grid_[0] - self.drop, *grid_[1:])]


def stale_lock(tile, drop):
    """Locks preset to 1 over NaN-poisoned partials. With drop = 0 every launched
    peer still stores before its owner reads (reversed order: a peer's head piece
    runs first, the owner's tail piece last), so this only reports. With drop > 0
    the dropped programs' partials are never stored and the result must be wrong."""
    bm, bn, bk = tile
    config = config_for(*tile)
    _, M, N, K, G, _ = cases(*tile)[2]
    a, w, bias, backing, ref = make_case(M, N, K, bm, bn, False)
    dev = torch.device("cuda", torch.cuda.current_device())
    mk._persistent_partials(dev, torch.cuda.current_stream(), G).fill_(float("nan"))
    sync_words = mk._sync_words(G, NUM_XCDS)
    flags(G)[sync_words: sync_words + G] = 1
    kernel = mk._gemm_a16w16_compute_bound_kernel
    if drop:
        mk._gemm_a16w16_compute_bound_kernel = DropPrograms(kernel, drop)
    try:
        run(a, w, bias, backing[:M, :N], config, G, True)
        torch.cuda.synchronize()
    finally:
        mk._gemm_a16w16_compute_bound_kernel = kernel
    ok, _, err = verdict(backing[:M, :N], ref, backing, M, N)
    flags(G).zero_()
    print(f"stale-lock drop={drop}: correct={ok} error={err}")
    if drop and ok:
        raise AssertionError("dropped-program negative control unexpectedly passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile", default="256x256x64")
    ap.add_argument("--launches", type=int, default=3)
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--stale-lock", action="store_true")
    ap.add_argument("--persistent-partials", action="store_true")
    args = ap.parse_args()
    tile = tuple(int(x) for x in args.tile.split("x"))
    if args.graph:
        return graph_replay(tile)
    if args.stale_lock:
        stale_lock(tile, 0)
        stale_lock(tile, NUM_XCDS)
        print("negative control failed as expected")
        return
    check_rejects(tile)
    check_cases(tile, args.launches, args.persistent_partials)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
