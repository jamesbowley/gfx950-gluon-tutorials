"""Experimental shape-dependent tile and Stream-K policy for v30.

Nothing in this module is required by the device kernel.  The constants and
formulas are provisional v29-derived choices, not an optimal policy, and need
more extensive tuning.  Some choices are retained only for consistency with
earlier versions, including the default tile for M < 512.  The policy is
intentionally isolated so it can be replaced or tuned without changing launch
data structures.
"""

import math
from typing import Mapping

from .selection import (
    DeviceConfig,
    StreamKConfig,
    TileSpec,
    cdiv,
    resolve_config as resolve_exact_config,
    xcd_local_split,
)


DEFAULT_TILE = TileSpec(256, 256, 64)
AUTO_TILES = (
    DEFAULT_TILE,
    TileSpec(128, 256, 64),
    TileSpec(256, 128, 64),
    TileSpec(128, 128, 128),
    TileSpec(128, 128, 64),
    TileSpec(64, 256, 64),
    TileSpec(256, 64, 64),
    TileSpec(64, 128, 128),
    TileSpec(128, 64, 128),
    TileSpec(64, 128, 64),
    TileSpec(128, 64, 64),
    TileSpec(64, 64, 128),
    TileSpec(64, 64, 64),
)

# Current empirical gfx950 thresholds.  AUTO_MIN_TILES counts output tiles
# before Stream-K policy or splitting is considered.
AUTO_MIN_TILES = 128
AUTO_HALVE_K = 4096


def tile_count(M, N, tile):
    return cdiv(M, tile.block_m) * cdiv(N, tile.block_n)


def group_size_m(block_m, block_n, grid_m, grid_mn, device):
    """Choose the existing footprint-minimizing tile group offline."""
    programs_per_xcd = max(
        1,
        min(device.num_programs, grid_mn) // device.num_xcds,
    )
    best = None
    for group_m in (1, 2, 4, 8, 16, 32, 64, 128):
        if group_m > max(1, grid_m):
            break
        footprint = (
            group_m * block_m
            + math.ceil(programs_per_xcd / group_m) * block_n
        )
        if best is None or footprint < best[0]:
            best = (footprint, group_m)
    return best[1]


def auto_tile(M, N, K, device):
    """Apply the current v29 quadrant-tile rule."""
    if not isinstance(device, DeviceConfig):
        raise TypeError("device must be a DeviceConfig")

    count = lambda tile: tile_count(M, N, tile)
    if M < 512 or count(DEFAULT_TILE) >= device.num_programs:
        return DEFAULT_TILE

    candidates = [tile for tile in AUTO_TILES if count(tile) >= AUTO_MIN_TILES]
    if not candidates:
        return TileSpec(64, 64, 128)

    biggest = max(tile.area for tile in candidates)
    pick = min(
        (tile for tile in candidates if tile.area == biggest),
        key=lambda tile: (count(tile), tile.block_k != 128),
    )
    if count(pick) == AUTO_MIN_TILES and K <= AUTO_HALVE_K:
        halves = [
            tile
            for tile in AUTO_TILES
            if 2 * tile.area == pick.area and count(tile) == device.num_programs
        ]
        if halves:
            pick = min(halves, key=lambda tile: tile.block_k != 128)
    return pick


def capped_policy(total_tiles, iters_per_tile, num_programs):
    """Return ``(policy, fixup, split_n)`` using v29's capped thresholds."""
    s = total_tiles % num_programs
    if s == 0:
        return "dp", "owner", None
    if 2 * s > num_programs:
        policy = "spread" if (num_programs - s) * iters_per_tile >= 10493 else "dp"
        return policy, "owner", None
    if total_tiles < num_programs:
        if (num_programs - s) * iters_per_tile < 14 * num_programs:
            return "dp", "owner", None
        cap = min(num_programs // s, iters_per_tile // 2)
        if num_programs // s >= 4:
            fixup, n = (
                "rs",
                min(cap, (3 * num_programs // 4) // s, round(iters_per_tile / 4)),
            )
        else:
            fixup, n = "owner", min(cap, round(math.sqrt(iters_per_tile / 1.5)))
    else:
        if (num_programs - s) * iters_per_tile < 20 * num_programs:
            return "dp", "owner", None
        if num_programs // s >= 8:
            fixup = "rs"
            n = min(num_programs // 2 // s, max(8, iters_per_tile // 8))
        else:
            fixup = "owner"
            n = min(
                num_programs // s,
                round(math.sqrt(iters_per_tile / max(2.2, s / 5))),
            )
    n = max(1, n)
    return ("dp", "owner", None) if n == 1 else ("one_tile", fixup, n)


def default_fixup(split_n):
    """Current explicit-policy crossing point in semantic split units."""
    return "rs" if split_n >= 8 else "owner"


def _reject(strict, message, error_type=ValueError):
    if strict:
        raise error_type(message)
    return None


def _candidate_fields(config):
    if isinstance(config, StreamKConfig):
        return {
            "block_m": config.block_m,
            "block_n": config.block_n,
            "block_k": config.block_k,
            "group_size_m": config.group_size_m,
            "policy": config.policy,
            "fixup": config.fixup,
            "split_n": config.split_n,
            "exchange": config.exchange,
        }
    if not isinstance(config, Mapping):
        raise TypeError("heuristic config must be a mapping")
    aliases = {
        "BLOCK_M": "block_m",
        "BLOCK_N": "block_n",
        "BLOCK_K": "block_k",
        "GROUP_SIZE_M": "group_size_m",
        "STREAMK_POLICY": "policy",
        "STREAMK_FIXUP": "fixup",
        "SPLIT_N": "split_n",
        "STREAMK_EXCHANGE": "exchange",
        "block_m": "block_m",
        "block_n": "block_n",
        "block_k": "block_k",
        "group_size_m": "group_size_m",
        "policy": "policy",
        "fixup": "fixup",
        "split_n": "split_n",
        "exchange": "exchange",
    }
    parsed = {}
    for key, value in config.items():
        try:
            field = aliases[key]
        except KeyError as error:
            raise ValueError(f"unknown Stream-K config key {key!r}") from error
        if field in parsed and parsed[field] != value:
            raise ValueError(f"conflicting values for Stream-K field {field!r}")
        parsed[field] = value
    return {
        "block_m": parsed.get("block_m"),
        "block_n": parsed.get("block_n"),
        "block_k": parsed.get("block_k"),
        "group_size_m": parsed.get("group_size_m"),
        "policy": parsed.get("policy", "capped"),
        "fixup": parsed.get("fixup", "auto"),
        "split_n": parsed.get("split_n"),
        "exchange": parsed.get("exchange", "wt"),
    }


def resolve_config(M, N, K, *, config, device, strict=True):
    """Complete optional fields and return an exact primitive configuration.

    Automatic fields may be derived. Explicit values are never clamped or
    silently changed; an infeasible explicit candidate is rejected.
    """
    try:
        fields = _candidate_fields(config)
    except (TypeError, ValueError):
        if strict:
            raise
        return None
    if not isinstance(device, DeviceConfig):
        return _reject(strict, "device must be a DeviceConfig", TypeError)
    if any(type(value) is not int for value in (M, N, K)):
        return _reject(strict, "M, N and K must be integers", TypeError)
    if min(M, N, K) <= 0:
        return _reject(strict, "M, N and K must be positive")

    blocks = (fields["block_m"], fields["block_n"], fields["block_k"])
    if any(value is None for value in blocks):
        if not all(value is None for value in blocks):
            return _reject(
                strict,
                "BLOCK_M, BLOCK_N and BLOCK_K must be supplied together",
            )
        chosen_tile = auto_tile(M, N, K, device)
    else:
        try:
            chosen_tile = TileSpec(*blocks)
        except (TypeError, ValueError):
            if strict:
                raise
            return None

    num_programs = device.num_programs
    total_tiles = tile_count(M, N, chosen_tile)
    k_pad = cdiv(K, 2 * chosen_tile.block_k) * 2 * chosen_tile.block_k
    iters_per_tile = k_pad // chosen_tile.block_k
    pairs = k_pad // (2 * chosen_tile.block_k)
    policy = fields["policy"]
    fixup = fields["fixup"]
    split_n = fields["split_n"]
    exchange = fields["exchange"]

    if policy == "capped":
        if split_n is not None or fixup != "auto" or exchange != "wt":
            return _reject(
                strict,
                "capped policy requires auto fixup, no split, and WT exchange",
            )
        policy, fixup, split_n = capped_policy(
            total_tiles,
            iters_per_tile,
            num_programs,
        )
    elif policy == "one_tile":
        leftover = total_tiles % num_programs
        if leftover == 0:
            return _reject(strict, "one_tile requires leftover tiles")
        split_cap = min(num_programs // leftover, pairs)
        if split_cap < 2:
            return _reject(
                strict,
                "shape leaves fewer than two programs/pairs per Stream-K tile",
            )
        if split_n is None:
            split_n = (
                xcd_local_split(leftover, split_cap, pairs, device)
                if exchange == "l2"
                else split_cap
            )
        if fixup == "auto":
            fixup = "rs" if exchange == "l2" else default_fixup(split_n)
    elif fixup == "auto":
        fixup = "owner"

    group_m = fields["group_size_m"]
    if group_m is None:
        group_m = group_size_m(
            chosen_tile.block_m,
            chosen_tile.block_n,
            cdiv(M, chosen_tile.block_m),
            total_tiles,
            device,
        )

    try:
        completed = StreamKConfig(
            block_m=chosen_tile.block_m,
            block_n=chosen_tile.block_n,
            block_k=chosen_tile.block_k,
            group_size_m=group_m,
            policy=policy,
            fixup=fixup,
            split_n=split_n,
            exchange=exchange,
        )
    except (TypeError, ValueError):
        if strict:
            raise
        return None
    if resolve_exact_config(
        M,
        N,
        K,
        config=completed,
        device=device,
        strict=strict,
    ) is None:
        return None
    return completed.to_dict()


def select_plan(
    M,
    N,
    K,
    *,
    device,
    tile,
    policy="capped",
    fixup="auto",
    split_n=None,
    exchange="wt",
):
    """Choose one complete primitive configuration offline.

    ``tile=None`` requests the experimental auto-tile rule.  Otherwise callers
    pass an integer :class:`TileSpec`; strings are deliberately not accepted.
    """
    if not isinstance(device, DeviceConfig):
        raise TypeError("device must be a DeviceConfig")
    if tile is not None and not isinstance(tile, TileSpec):
        raise TypeError("tile must be a TileSpec or None for auto selection")

    config = {
        "BLOCK_M": None if tile is None else tile.block_m,
        "BLOCK_N": None if tile is None else tile.block_n,
        "BLOCK_K": None if tile is None else tile.block_k,
        "STREAMK_POLICY": policy,
        "STREAMK_FIXUP": fixup,
        "SPLIT_N": split_n,
        "STREAMK_EXCHANGE": exchange,
    }
    return resolve_config(M, N, K, config=config, device=device, strict=True)
