"""Quadrant-only correctness and synchronization checks for v30.

Run from ``a16w16`` after sourcing the usual gfx950 environment:

    python v30_streamk_streamlined/check_streamk.py --tile 64x64x64

The stale-count mode is a negative control and should fail or time out:

    timeout 120 python v30_streamk_streamlined/check_streamk.py \
        --tile 64x64x64 --stale-count

``--graph`` replays captured L2 launches. ``--stale-flag`` presets the L2
flags and must give a wrong result (use 128x128x128; smaller skews hide it).
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import bench_ext as bench  # noqa: E402  (adds kernels/gemm/utils to sys.path)
import torch  # noqa: E402
from v30_streamk_streamlined.hardware import query_device_config  # noqa: E402
from v30_streamk_streamlined.heuristics import select_plan  # noqa: E402
from v30_streamk_streamlined.matmul_kernel import gemm_a16w16  # noqa: E402
from v30_streamk_streamlined import runtime  # noqa: E402
from v30_streamk_streamlined.runtime import streamk_workspace  # noqa: E402
from v30_streamk_streamlined.selection import (  # noqa: E402
    TileSpec,
    resolve_config,
)

SENTINEL = 1234.0


def case_shapes(tile):
    bm, bn, bk = tile.block_m, tile.block_n, tile.block_k
    k = 8 * bk
    return {
        "dp": (16 * bm, 16 * bn, k),
        "tail": (17 * bm, 16 * bn, k),
        "capped": (17 * bm, 16 * bn, max(k, 8192)),
        "spread": (16 * bm, 28 * bn, k),
        "compact": (3 * bm, 11 * bn, k),
        "ragged": (17 * bm - bm // 8, 16 * bn - bn // 8, k - 16),
        "ragged_m": (17 * bm - bm // 8, 16 * bn, k),
        "ragged_n": (17 * bm, 16 * bn - bn // 8, k),
        # K_RES > BLOCK_K (K-step 0 of the last pair masked too) and K_RES ==
        # BLOCK_K (K-step 1 all padding, step 0 unmasked).
        "kres_gt": (17 * bm, 16 * bn, k - bk - 16),
        "kres_eq": (17 * bm, 16 * bn, k - bk),
    }


def configurations(tile):
    shapes = case_shapes(tile)
    return [
        ("dp", shapes["dp"], dict(policy="dp"), False),
        ("capped", shapes["capped"], dict(policy="capped"), False),
        (
            "owner",
            shapes["tail"],
            dict(policy="one_tile", split_n=4, fixup="owner"),
            False,
        ),
        (
            "rs",
            shapes["tail"],
            dict(policy="one_tile", split_n=4, fixup="rs"),
            True,
        ),
        (
            "l2",
            shapes["tail"],
            dict(policy="one_tile", split_n=4, fixup="rs", exchange="l2"),
            False,
        ),
        ("two_tile", shapes["tail"], dict(policy="two_tile"), False),
        ("spread", shapes["spread"], dict(policy="spread"), False),
        (
            "compact",
            shapes["compact"],
            dict(policy="one_tile", split_n=2, fixup="owner"),
            False,
        ),
        (
            "ragged",
            shapes["ragged"],
            dict(policy="one_tile", split_n=2, fixup="rs"),
            True,
        ),
        (
            "ragged_m",
            shapes["ragged_m"],
            dict(policy="one_tile", split_n=4, fixup="rs"),
            True,
        ),
        (
            "ragged_n",
            shapes["ragged_n"],
            dict(policy="one_tile", split_n=4, fixup="owner"),
            False,
        ),
        ("dp_below_wave", shapes["compact"], dict(policy="dp"), False),
        ("kres_gt_dp", shapes["kres_gt"], dict(policy="dp"), True),
        (
            "kres_gt_rs",
            shapes["kres_gt"],
            dict(policy="one_tile", split_n=4, fixup="rs"),
            False,
        ),
        ("kres_eq_dp", shapes["kres_eq"], dict(policy="dp"), False),
        (
            "kres_eq_owner",
            shapes["kres_eq"],
            dict(policy="one_tile", split_n=4, fixup="owner"),
            True,
        ),
    ]


def extra_configurations(tile):
    """Persistent phases of one and several trips, before no Stream-K tail and before
    each kind of tail: the last trip's prefetch of the first Stream-K pair."""
    bm, bn, bk = tile.block_m, tile.block_n, tile.block_k
    k = 8 * bk
    return [
        ("dp_1wave", (16 * bm, 16 * bn, k), dict(policy="dp"), False),
        ("dp_4wave", (32 * bm, 32 * bn, k), dict(policy="dp"), True),
        ("dp_4w_kres", (32 * bm, 32 * bn, k - bk - 16), dict(policy="dp"), False),
        (
            "rs_3wave",
            (49 * bm, 16 * bn, k),
            dict(policy="one_tile", split_n=4, fixup="rs"),
            False,
        ),
        (
            "rs_3w_kres",
            (49 * bm, 16 * bn, k - bk - 16),
            dict(policy="one_tile", split_n=4, fixup="rs"),
            True,
        ),
        (
            "l2_3w_kres",
            (49 * bm, 16 * bn, k - bk),
            dict(policy="one_tile", split_n=4, fixup="rs", exchange="l2"),
            False,
        ),
        ("two_tile_2w", (33 * bm, 16 * bn, k), dict(policy="two_tile"), False),
        (
            "two_tile_kres",
            (33 * bm - bm // 8, 16 * bn, k - bk - 16),
            dict(policy="two_tile"),
            True,
        ),
        ("spread_2w", (16 * bm, 44 * bn, k), dict(policy="spread"), False),
    ]


def make_case(M, N, K, tile, with_bias):
    torch.manual_seed(17)
    a = torch.rand((M, K), device=bench.DEVICE, dtype=torch.bfloat16) - 0.5
    weights = torch.rand((N, K), device=bench.DEVICE, dtype=torch.bfloat16) - 0.5
    bias = (
        torch.rand((N,), device=bench.DEVICE, dtype=torch.bfloat16) - 0.5
        if with_bias
        else None
    )
    # A padded output window detects stores outside logical C.
    backing = torch.full(
        (M + tile.block_m, N + tile.block_n),
        SENTINEL,
        device=bench.DEVICE,
        dtype=torch.bfloat16,
    )
    c = backing[:M, :N]
    ref = a.float() @ weights.float().T
    if bias is not None:
        ref += bias.float()
    return a, weights, c, backing, bias, ref.to(torch.bfloat16)


def guards_intact(backing, M, N):
    return bool(
        torch.all(backing[M:, :] == SENTINEL)
        and torch.all(backing[:M, N:] == SENTINEL)
    )


PERSISTENT = False
STREAMS = []


def gemm(a, weights, bias, c, config):
    stream = torch.cuda.current_stream()
    if stream not in STREAMS:
        STREAMS.append(stream)
    gemm_a16w16(a, weights, bias, y=c, config=config, persistent_partials=PERSISTENT)


def synchronization_rearmed(device):
    """Every used stream's counts, L2 flags and locks are back at zero;
    generation words only increment and are skipped."""
    torch.cuda.synchronize()
    P = device.num_programs
    index = torch.cuda.current_device()
    for stream in STREAMS:
        flags = runtime._flags(
            torch.device("cuda", index), stream, P, device.num_xcds
        )
        if torch.count_nonzero(flags[: 2 * P : 2]) or torch.count_nonzero(
            flags[2 * P :]
        ):
            return False
    return True


def persistent_partials(tile, device):
    """The persistent partial buffers of every used stream (empty when off)."""
    if not PERSISTENT:
        return []
    index = torch.cuda.current_device()
    return [
        runtime._persistent_partials(
            torch.device("cuda", index), stream, device.num_programs
        )
        for stream in STREAMS
    ]


def launch_workspace(M, N, K, tile, device, config):
    launch = resolve_config(M, N, K, config=config, device=device)
    return streamk_workspace(
        bench.DEVICE,
        device.num_programs,
        device.num_xcds,
        tile.block_m,
        tile.block_n,
        launch["STREAMK_PROGRAMS"],
        config["STREAMK_EXCHANGE"] == "l2",
        PERSISTENT,
    )


def run_case(label, shape, options, with_bias, tile, device, launches):
    M, N, K = shape
    config = select_plan(M, N, K, device=device, tile=tile, **options)
    launch = resolve_config(M, N, K, config=config, device=device)
    a, weights, c, backing, bias, ref = make_case(M, N, K, tile, with_bias)
    for _ in range(launches):
        backing.fill_(SENTINEL)
        gemm(a, weights, bias, c, config)
    torch.cuda.synchronize()
    correct = torch.allclose(c, ref, atol=0.1, rtol=0.01)
    guard = guards_intact(backing, M, N)
    rearmed = synchronization_rearmed(device)
    error = torch.max(torch.abs(c.float() - ref.float())).item()
    print(
        f"{label:10s} {M}x{N}x{K} "
        f"{launch['STREAMK_POLICY']}/{launch['STREAMK_FIXUP']} "
        f"S={launch['STREAMK_TILES']} n={launch['RESOLVED_SPLIT_N']} "
        f"programs={launch['STREAMK_PROGRAMS']} grid={launch['GRID']} "
        f"error={error:.5f} correct={correct} guard={guard} rearmed={rearmed}"
    )
    if not (correct and guard and rearmed):
        raise AssertionError(f"{label} failed")


def stale_count(tile, device):
    M, N, K = case_shapes(tile)["tail"]
    config = select_plan(
        M,
        N,
        K,
        device=device,
        tile=tile,
        policy="one_tile",
        split_n=2,
        fixup="rs",
    )
    a, weights, c, _, bias, ref = make_case(M, N, K, tile, False)
    _, locks, sync = launch_workspace(M, N, K, tile, device, config)
    locks.zero_()
    sync.zero_()
    # A stale count equal to n means no arriver observes n - 1, so readers can
    # never observe a generation bump. The external timeout must terminate it.
    sync[0] = 2
    gemm(a, weights, bias, c, config)
    torch.cuda.synchronize()
    if torch.allclose(c, ref, atol=0.1, rtol=0.01):
        raise AssertionError("stale-count negative control unexpectedly passed")
    print("stale-count failed as expected")


def l2_cases(tile, device):
    """Two L2 configurations: one tile at the largest XCD-local split, and
    the 17-wave tail at split 4."""
    K = 8192
    split_n = min(
        32,
        K // (2 * tile.block_k),
        device.num_programs // device.num_xcds,
    )
    cases = []
    for (M, N, K), n in (
        ((tile.block_m, tile.block_n, K), split_n),
        (case_shapes(tile)["tail"], 4),
    ):
        config = select_plan(
            M,
            N,
            K,
            device=device,
            tile=tile,
            policy="one_tile",
            split_n=n,
            fixup="rs",
            exchange="l2",
        )
        cases.append((config, make_case(M, N, K, tile, False)))
    return cases


def check_launch(label, case):
    a, weights, c, backing, _, _ = case
    M, N = c.shape
    ref = (a.float() @ weights.float().T).to(torch.bfloat16)
    correct = torch.allclose(c, ref, atol=0.1, rtol=0.01)
    guard = guards_intact(backing, M, N)
    if not (correct and guard):
        error = torch.max(torch.abs(c.float() - ref.float())).item()
        raise AssertionError(f"{label} failed: error={error:.5f} guard={guard}")


def graph_replay(tile, device, replays=20):
    """Replay captured L2 launches over changing inputs, so a barrier that
    passes early reads the previous replay's partials, or NaN when persistent
    partials are poisoned. The warm-up keeps the flag zero fill out of the
    graph."""
    cases = l2_cases(tile, device)

    def launch(index):
        config, (a, weights, c, _, bias, _) = cases[index]
        gemm(a, weights, bias, c, config)

    capture = torch.cuda.Stream()
    capture.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture):
        for index in range(len(cases)):
            launch(index)
    torch.cuda.synchronize()

    def refresh():
        for _, case in cases:
            case[0].uniform_(-0.5, 0.5)
            case[3].fill_(SENTINEL)
        for partials in persistent_partials(tile, device):
            partials.fill_(float("nan"))

    single = torch.cuda.CUDAGraph()
    with torch.cuda.graph(single, stream=capture):
        launch(0)
    pair = torch.cuda.CUDAGraph()
    with torch.cuda.graph(pair, stream=capture):
        launch(0)
        launch(1)
    torch.cuda.synchronize()
    if not synchronization_rearmed(device):
        raise AssertionError("graph capture left synchronization armed")

    for replay in range(replays):
        refresh()
        single.replay()
        torch.cuda.synchronize()
        check_launch(f"single-node replay {replay}", cases[0][1])

        refresh()
        pair.replay()
        torch.cuda.synchronize()
        for index, (_, case) in enumerate(cases):
            check_launch(f"two-node replay {replay} node {index}", case)

        refresh()
        launch(replay % 2)
        single.replay()
        torch.cuda.synchronize()
        check_launch(f"eager then replay {replay}", cases[0][1])
        if replay % 2:
            check_launch(f"eager {replay}", cases[1][1])
        if not synchronization_rearmed(device):
            raise AssertionError(f"replay {replay} left synchronization armed")
    print(f"graph replay passed: {replays} rounds of single, two-node and mixed")


def stale_flag(tile, device):
    """Reader rows left at 1 must let a reader pass before its peers. Three
    pairs over two programs: reader 1 runs one pair and reads chunk 0's
    partial a whole pair before chunk 0 stores it."""
    M, N, K = tile.block_m, tile.block_n, 6 * tile.block_k
    config = select_plan(
        M,
        N,
        K,
        device=device,
        tile=tile,
        policy="one_tile",
        split_n=2,
        fixup="rs",
        exchange="l2",
    )
    case = make_case(M, N, K, tile, False)
    a, weights, c, _, bias, _ = case
    gemm(a, weights, bias, c, config)
    torch.cuda.synchronize()
    # New inputs make any early-read partial stale, even when a per-call
    # allocation reuses the previous launch's block.
    a.uniform_(-0.5, 0.5)
    for partials in persistent_partials(tile, device):
        partials.fill_(float("nan"))
    _, _, sync = launch_workspace(M, N, K, tile, device, config)
    sync[2 * device.num_programs :] = 1
    gemm(a, weights, bias, c, config)
    torch.cuda.synchronize()
    try:
        check_launch("stale-flag", case)
    except AssertionError:
        print("stale-flag failed as expected")
        return
    raise AssertionError("stale-flag negative control unexpectedly passed")


def concurrent_streams(tile, device, rounds=20):
    """Different Stream-K configurations launched concurrently on two streams
    must not share synchronization or partials."""
    pairs = []
    for label in ("rs", "owner"):
        _, shape, options, _ = next(c for c in configurations(tile) if c[0] == label)
        config = select_plan(*shape, device=device, tile=tile, **options)
        pairs.append((config, make_case(*shape, tile, False), torch.cuda.Stream()))
    torch.cuda.synchronize()
    for round_ in range(rounds):
        for config, (a, weights, c, backing, bias, _), stream in pairs:
            backing.fill_(SENTINEL)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(4):
                    gemm(a, weights, bias, c, config)
        torch.cuda.synchronize()
        for index, (_, case, _) in enumerate(pairs):
            check_launch(f"stream {index} round {round_}", case)
    if not synchronization_rearmed(device):
        raise AssertionError("concurrent streams left synchronization armed")
    print(f"concurrent streams passed: {rounds} rounds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile", default="64x64x64")
    parser.add_argument("--launches", type=int, default=3)
    parser.add_argument("--stale-count", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--stale-flag", action="store_true")
    parser.add_argument("--streams", action="store_true")
    parser.add_argument("--persistent-partials", action="store_true")
    parser.add_argument("--extra", action="store_true")
    args = parser.parse_args()
    global PERSISTENT
    PERSISTENT = args.persistent_partials
    try:
        tile = TileSpec(*(int(part) for part in args.tile.split("x")))
    except (TypeError, ValueError) as error:
        parser.error(f"--tile must be BMxBNxBK: {error}")
    device = query_device_config()
    if args.stale_count:
        stale_count(tile, device)
        return
    if args.graph:
        graph_replay(tile, device)
        return
    if args.stale_flag:
        stale_flag(tile, device)
        return
    if args.streams:
        concurrent_streams(tile, device)
        return
    for case in extra_configurations(tile) if args.extra else configurations(tile):
        run_case(*case, tile, device, args.launches)


if __name__ == "__main__":
    main()
