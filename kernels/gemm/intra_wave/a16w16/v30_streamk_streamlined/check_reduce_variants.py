"""Correctness checks for reduce_variants.py (WORKLOG entry 15).

Run from ``a16w16`` after sourcing the usual gfx950 environment:

    python v30_streamk_streamlined/check_reduce_variants.py --tile 64x64x64
    python v30_streamk_streamlined/check_reduce_variants.py --tile 64x64x64 --graph
    python v30_streamk_streamlined/check_reduce_variants.py --tile 128x128x128 --stale-lock

Every launch uses the persistent partial buffer, poisoned with NaN first, so a
reader that passes early reads NaN. Each case checks the result against torch,
C's sentinel guards, and that the locks and barrier counts are back at zero after
three launches; rs and rs_all must also be bitwise equal to production v30.
``--stale-lock`` presets every lock to 1 before a tree launch and must give a
wrong result. ``--stale-count`` presets an rs_all barrier count and must hang
(run it under ``timeout``) or give a wrong result.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import bench_ext as bench  # noqa: E402,F401  (adds kernels/gemm/utils to sys.path)
import torch  # noqa: E402
from v30_streamk_streamlined import runtime  # noqa: E402
from v30_streamk_streamlined.check_streamk import (  # noqa: E402
    SENTINEL,
    case_shapes,
    guards_intact,
    make_case,
)
from v30_streamk_streamlined.hardware import query_device_config  # noqa: E402
from v30_streamk_streamlined.heuristics import select_plan  # noqa: E402
from v30_streamk_streamlined.matmul_kernel import gemm_a16w16  # noqa: E402
from v30_streamk_streamlined.reduce_variants import gemm_a16w16_variant  # noqa: E402
from v30_streamk_streamlined.selection import TileSpec, resolve_config  # noqa: E402

# (label, variant, tree radix, option: tree_static, "sleep" or "packed")
VARIANTS = [
    ("rs", "rs", 2, None),
    ("tree2", "tree", 2, None),
    ("tree4", "tree", 4, None),
    ("tree2 loop", "tree", 2, False),
    ("tree4 loop", "tree", 4, False),
    ("separate", "separate", 2, None),
    ("rs_all", "rs_all", 2, None),
    ("rs_all slp", "rs_all", 2, "sleep"),
    ("rs_all pkd", "rs_all", 2, "packed"),
]


def cases(tile, device):
    """(label, shape, split_n, bias): the tail at splits 2-4, the long-K tail
    (capped's shape) at 5, 8 and 16, and each other shape at split 2-4."""
    shapes = case_shapes(tile)
    out = []
    for n in (2, 3, 4):
        out.append((f"tail n={n}", shapes["tail"], n, n % 2 == 1))
    for n in (5, 8, 16):
        out.append((f"longk n={n}", shapes["capped"], n, n == 8))
    for label, n, bias in (
        ("compact", 2, True),
        ("ragged", 2, True),
        ("ragged_m", 4, True),
        ("ragged_n", 4, False),
        ("kres_gt", 4, True),
        ("kres_eq", 3, False),
    ):
        out.append((label, shapes[label], n, bias))
    valid = []
    for label, shape, n, bias in out:
        config = plan(shape, tile, device, n)
        if config is not None:
            valid.append((label, shape, n, bias, config))
    return valid


def plan(shape, tile, device, n):
    M, N, K = shape
    try:
        config = select_plan(
            M, N, K, device=device, tile=tile, policy="one_tile", split_n=n, fixup="rs"
        )
        resolve_config(M, N, K, config=config, device=device)
    except ValueError:
        return None
    return config


def partials_buffer(device):
    index = torch.cuda.current_device()
    stream = torch.cuda.current_stream()
    return runtime._persistent_partials(
        torch.device("cuda", index), stream, device.num_programs
    )


def rearmed(device):
    torch.cuda.synchronize()
    P = device.num_programs
    index = torch.cuda.current_device()
    flags = runtime._flags(
        torch.device("cuda", index), torch.cuda.current_stream(), P, device.num_xcds
    )
    return not (
        torch.count_nonzero(flags[: 2 * P : 2]) or torch.count_nonzero(flags[2 * P :])
    )


def run(variant, radix, a, weights, bias, c, config, option=None):
    """option: tree_static for the tree; "sleep" (sleeping poll) or "packed" (barriers
    at rs's slots) for rs_all."""
    gemm_a16w16_variant(
        a,
        weights,
        bias,
        y=c,
        config=config,
        variant=variant,
        tree_radix=radix,
        tree_static=option if variant == "tree" else None,
        poll_sleep=int(option == "sleep"),
        spread_barriers=False if option == "packed" else None,
        persistent_partials=True,
    )


def run_case(label, shape, n, with_bias, config, tile, device, launches):
    M, N, K = shape
    a, weights, c, backing, bias, ref = make_case(M, N, K, tile, with_bias)
    prod = torch.empty_like(c)
    gemm_a16w16(a, weights, bias, y=prod, config=config)
    variants = list(VARIANTS)
    if n == 16:
        variants.append(("tree8", "tree", 8, None))  # 14 children: the run-time loop
    for name, variant, radix, static in variants:
        for _ in range(launches):
            partials_buffer(device).fill_(float("nan"))
            backing.fill_(SENTINEL)
            run(variant, radix, a, weights, bias, c, config, static)
        torch.cuda.synchronize()
        correct = torch.allclose(c, ref, atol=0.1, rtol=0.01)
        guard = guards_intact(backing, M, N)
        armed = rearmed(device)
        # Every reader count sums a block's partials in the same order, so the old
        # rs, rs_all and production (rs_all since entry 18) agree bit for bit.
        bitwise = name not in ("rs", "rs_all") or torch.equal(c, prod)
        error = torch.max(torch.abs(c.float() - ref.float())).item()
        print(
            f"{label:12s} {M}x{N}x{K} bias={with_bias:d} {name:10s} "
            f"error={error:.5f} correct={correct} guard={guard} rearmed={armed} "
            f"bitwise_prod={bitwise}"
        )
        if not (correct and guard and armed and bitwise):
            raise AssertionError(f"{label} {name} failed")


def graph_replay(tile, device, replays=20):
    """Capture tree2, tree4 and separate on the tail at split 4 and replay them over
    new inputs with poisoned partials."""
    shape = case_shapes(tile)["tail"]
    config = plan(shape, tile, device, 4)
    a, weights, c, backing, bias, _ = make_case(*shape, tile, True)
    for name, variant, radix, static in VARIANTS[1:]:
        capture = torch.cuda.Stream()
        capture.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture):
            run(variant, radix, a, weights, bias, c, config, static)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=capture):
            run(variant, radix, a, weights, bias, c, config, static)
            run(variant, radix, a, weights, bias, c, config, static)
        torch.cuda.synchronize()
        with torch.cuda.stream(capture):
            for replay in range(replays):
                a.uniform_(-0.5, 0.5)
                backing.fill_(SENTINEL)
                partials_buffer(device).fill_(float("nan"))
                g.replay()
                capture.synchronize()
                ref = a.float() @ weights.float().T + bias.float()
                ok = torch.allclose(c, ref.to(c.dtype), atol=0.1, rtol=0.01)
                if not (ok and guards_intact(backing, *c.shape) and rearmed(device)):
                    raise AssertionError(f"{name} replay {replay} failed")
        print(f"{name}: graph replay passed, {replays} rounds of a two-launch graph")


def stale_lock(tile, device):
    shape = case_shapes(tile)["tail"]
    config = plan(shape, tile, device, 4)
    a, weights, c, backing, bias, ref = make_case(*shape, tile, False)
    run("tree", 2, a, weights, bias, c, config)
    torch.cuda.synchronize()
    a.uniform_(-0.5, 0.5)
    ref = (a.float() @ weights.float().T).to(c.dtype)
    partials_buffer(device).fill_(float("nan"))
    index = torch.cuda.current_device()
    flags = runtime._flags(
        torch.device("cuda", index),
        torch.cuda.current_stream(),
        device.num_programs,
        device.num_xcds,
    )
    flags[runtime._sync_words(device.num_programs, device.num_xcds) :] = 1
    run("tree", 2, a, weights, bias, c, config)
    torch.cuda.synchronize()
    flags.zero_()
    if torch.allclose(c, ref, atol=0.1, rtol=0.01):
        raise AssertionError("stale-lock negative control unexpectedly passed")
    print("stale-lock failed as expected")


def stale_count(tile, device):
    """rs_all with tile 0's barrier count preset to its n + E arrivals: no arriver
    sees the last count, so its readers never pass. Run under a timeout."""
    from v30_streamk_streamlined.reduce_variants import rs_all_readers

    shape = case_shapes(tile)["tail"]
    config = plan(shape, tile, device, 2)
    a, weights, c, _, bias, ref = make_case(*shape, tile, False)
    S = resolve_config(*shape, config=config, device=device)["STREAMK_TILES"]
    _, _, _, E = rs_all_readers(tile, S, 2, device.num_programs)
    index = torch.cuda.current_device()
    flags = runtime._flags(
        torch.device("cuda", index),
        torch.cuda.current_stream(),
        device.num_programs,
        device.num_xcds,
    )
    flags[0] = 2 + E
    run("rs_all", 2, a, weights, bias, c, config)
    torch.cuda.synchronize()
    if torch.allclose(c, ref, atol=0.1, rtol=0.01):
        raise AssertionError("stale-count negative control unexpectedly passed")
    print("stale-count failed as expected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile", default="64x64x64")
    parser.add_argument("--launches", type=int, default=3)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--stale-lock", action="store_true")
    parser.add_argument("--stale-count", action="store_true")
    args = parser.parse_args()
    tile = TileSpec(*(int(part) for part in args.tile.split("x")))
    device = query_device_config()
    if args.stale_count:
        stale_count(tile, device)
        return
    if args.graph:
        graph_replay(tile, device)
        return
    if args.stale_lock:
        stale_lock(tile, device)
        return
    for label, shape, n, bias, config in cases(tile, device):
        run_case(label, shape, n, bias, config, tile, device, args.launches)


if __name__ == "__main__":
    main()
