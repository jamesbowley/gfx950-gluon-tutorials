"""Exact configuration validation and mechanical launch constants for v30."""

from dataclasses import dataclass
from typing import Mapping

QUADRANT_SIDES = (64, 128, 256)


def cdiv(x, y):
    return (x + y - 1) // y


@dataclass(frozen=True)
class TileSpec:
    """Integer geometry of one supported quadrant tile."""

    block_m: int
    block_n: int
    block_k: int

    def __post_init__(self):
        if any(
            type(value) is not int
            for value in (self.block_m, self.block_n, self.block_k)
        ):
            raise TypeError("tile dimensions must be integers")
        if self.block_m not in QUADRANT_SIDES or self.block_n not in QUADRANT_SIDES:
            raise ValueError(
                "v30 deliberately excludes small tiles: "
                f"BM and BN must be in {QUADRANT_SIDES}, got {self.block_m}x{self.block_n}"
            )
        if self.block_k not in (64, 128):
            raise ValueError(f"v30 requires BK in (64, 128), got {self.block_k}")

    @property
    def spec(self):
        return f"{self.block_m}x{self.block_n}x{self.block_k}"

    @property
    def area(self):
        return self.block_m * self.block_n

    @property
    def blocks(self):
        """Number of 32x32 reduce-scatter blocks in the output tile."""
        return self.area // (32 * 32)

    @property
    def quadrant_blocks(self):
        """Number of 32x32 reduce-scatter blocks in one tile quadrant."""
        return self.blocks // 4

    def __str__(self):
        return self.spec


@dataclass(frozen=True)
class DeviceConfig:
    """Device-dependent constants required to derive and launch a plan."""

    gfx: str
    num_programs: int
    num_xcds: int
    lds_limit_bytes: int

    def __post_init__(self):
        if not self.gfx:
            raise ValueError("gfx must be non-empty")
        if any(
            type(value) is not int
            for value in (self.num_programs, self.num_xcds, self.lds_limit_bytes)
        ):
            raise TypeError("device counts and limits must be integers")
        if self.num_programs <= 0:
            raise ValueError("num_programs must be positive")
        if self.num_xcds <= 0:
            raise ValueError("num_xcds must be positive")
        if self.num_programs % self.num_xcds:
            raise ValueError("num_programs must be divisible by num_xcds")
        if self.lds_limit_bytes <= 0:
            raise ValueError("lds_limit_bytes must be positive")


@dataclass(frozen=True)
class StreamKConfig:
    """Complete flat configuration accepted at the AITER API boundary."""

    block_m: int
    block_n: int
    block_k: int
    group_size_m: int
    policy: str
    fixup: str
    split_n: int | None
    exchange: str

    def __post_init__(self):
        TileSpec(self.block_m, self.block_n, self.block_k)
        if type(self.group_size_m) is not int:
            raise TypeError("GROUP_SIZE_M must be an integer")
        if self.group_size_m <= 0:
            raise ValueError("GROUP_SIZE_M must be positive")
        if self.policy not in ("dp", "one_tile", "two_tile", "spread"):
            raise ValueError(f"invalid Stream-K policy {self.policy!r}")
        if self.fixup not in ("owner", "rs"):
            raise ValueError(f"invalid Stream-K fixup {self.fixup!r}")
        if self.exchange not in ("wt", "l2"):
            raise ValueError(f"invalid Stream-K exchange {self.exchange!r}")
        if self.split_n is not None:
            if type(self.split_n) is not int:
                raise TypeError("split_n must be an integer or None")
            if self.split_n < 2:
                raise ValueError("split_n must be at least 2")
        if self.policy == "one_tile" and self.split_n is None:
            raise ValueError("one_tile requires SPLIT_N")
        if self.policy != "one_tile" and self.split_n is not None:
            raise ValueError(f"{self.policy} does not take split_n")
        if self.policy != "one_tile" and self.fixup != "owner":
            raise ValueError(f"{self.policy} requires owner fixup")
        if self.exchange == "l2" and self.policy != "one_tile":
            raise ValueError("l2 exchange requires one_tile policy")
        if self.exchange == "l2" and self.fixup != "rs":
            raise ValueError("l2 exchange requires reduce-scatter fixup")

    @property
    def tile(self):
        return TileSpec(self.block_m, self.block_n, self.block_k)

    def to_dict(self):
        """Return the stable primitive-only dictionary used by AITER."""
        return {
            "BLOCK_M": self.block_m,
            "BLOCK_N": self.block_n,
            "BLOCK_K": self.block_k,
            "GROUP_SIZE_M": self.group_size_m,
            "STREAMK_POLICY": self.policy,
            "STREAMK_FIXUP": self.fixup,
            "SPLIT_N": self.split_n,
            "STREAMK_EXCHANGE": self.exchange,
        }

    @classmethod
    def from_dict(cls, values):
        """Parse an AITER config dictionary without accepting unknown knobs."""
        if not isinstance(values, Mapping):
            raise TypeError("Stream-K config must be a mapping")
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
        for key, value in values.items():
            try:
                field = aliases[key]
            except KeyError as error:
                raise ValueError(f"unknown Stream-K config key {key!r}") from error
            if field in parsed and parsed[field] != value:
                raise ValueError(f"conflicting values for Stream-K field {field!r}")
            parsed[field] = value
        required = {
            "block_m",
            "block_n",
            "block_k",
            "group_size_m",
            "policy",
            "fixup",
            "split_n",
            "exchange",
        }
        missing = sorted(required - parsed.keys())
        if missing:
            raise ValueError(
                "incomplete Stream-K config; missing " + ", ".join(missing)
            )
        return cls(**parsed)


@dataclass(frozen=True)
class ReduceScatterConfig:
    readers: int
    blocks_per_group: int
    partials_per_trip: int


def pow2_floor(x):
    return 1 << (x.bit_length() - 1)


def reduce_scatter_config(tile, split_n, loads_in_flight=16):
    readers = min(pow2_floor(split_n), tile.blocks)
    nb = max(
        1,
        min(
            tile.blocks // readers,
            loads_in_flight,
            tile.quadrant_blocks,
        ),
    )
    return ReduceScatterConfig(
        readers=readers,
        blocks_per_group=nb,
        partials_per_trip=max(loads_in_flight // nb, 1),
    )


def xcd_local_split(streamk_tiles, n_cap, pairs_per_tile, device):
    per_xcd_tiles = cdiv(streamk_tiles, device.num_xcds)
    return max(
        1,
        min(
            n_cap,
            pairs_per_tile,
            (device.num_programs // device.num_xcds) // per_xcd_tiles,
        ),
    )


def _reject(strict, message, error_type=ValueError):
    if strict:
        raise error_type(message)
    return None


def resolve_config(M, N, K, *, config, device, strict=True):
    """Validate an exact config and return primitive mechanical launch constants."""
    if not isinstance(config, StreamKConfig):
        try:
            config = StreamKConfig.from_dict(config)
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
    tile = config.tile

    num_programs = device.num_programs
    total_tiles = cdiv(M, tile.block_m) * cdiv(N, tile.block_n)
    k_pad = cdiv(K, 2 * tile.block_k) * 2 * tile.block_k
    pairs = k_pad // (2 * tile.block_k)
    leftover = total_tiles % num_programs

    streamk_tiles = 0 if config.policy == "dp" else leftover
    if config.policy == "two_tile":
        if leftover == 0 or total_tiles < num_programs:
            return _reject(
                strict,
                "two_tile requires a partial wave after at least one full wave",
            )
        streamk_tiles += num_programs
    if config.policy in ("one_tile", "spread") and leftover == 0:
        return _reject(strict, f"{config.policy} requires leftover tiles")
    if config.policy == "spread" and 2 * leftover <= num_programs:
        return _reject(
            strict,
            "spread requires more than half of the resident programs "
            "as leftover tiles",
        )

    end_to_end = config.policy in ("two_tile", "spread")
    l2 = config.exchange == "l2" and streamk_tiles > 0
    xcd_split = 1
    if config.policy == "one_tile":
        split_cap = min(num_programs // streamk_tiles, pairs)
        split_n = config.split_n
        if split_cap < 2:
            return _reject(
                strict,
                "shape leaves fewer than two programs/pairs per Stream-K tile",
            )
        if split_n > split_cap:
            return _reject(
                strict,
                f"split_n={split_n} exceeds this shape's maximum {split_cap}",
            )
        if l2:
            xcd_split = xcd_local_split(streamk_tiles, split_n, pairs, device)
            if xcd_split != split_n:
                return _reject(
                    strict,
                    f"split_n={split_n} cannot be placed within one XCD; "
                    f"maximum is {xcd_split}",
                )
        streamk_programs = streamk_tiles * split_n
    elif end_to_end:
        split_n = 1
        streamk_programs = num_programs
        if streamk_tiles * pairs < streamk_programs:
            return _reject(
                strict,
                f"{config.policy} has fewer pair units than the {num_programs} programs",
            )
    else:
        split_n = 1
        streamk_programs = 0

    compact = (
        streamk_tiles > 0
        and streamk_tiles == total_tiles
        and not l2
        and streamk_programs < num_programs
    )
    if compact:
        grid = streamk_programs
    elif streamk_tiles == 0:
        grid = min(num_programs, total_tiles)
    else:
        grid = num_programs
    rs = reduce_scatter_config(tile, split_n if config.policy == "one_tile" else 1)
    return {
        **config.to_dict(),
        "NUM_PROGRAMS": num_programs,
        "NUM_XCDS": device.num_xcds,
        "TOTAL_TILES": total_tiles,
        "STREAMK_TILES": streamk_tiles,
        "STREAMK_PROGRAMS": streamk_programs,
        "PAIRS_PER_TILE": pairs,
        "RESOLVED_SPLIT_N": split_n,
        "END_TO_END": end_to_end,
        "L2": l2,
        "XCD_SPLIT": xcd_split,
        "COMPACT": compact,
        "GRID": grid,
        "RS_NB": rs.blocks_per_group,
        "RS_U": rs.partials_per_trip,
        "RS_R": rs.readers,
    }


def try_resolve_config(M, N, K, *, config, device):
    """Return mechanical constants or ``None`` when an exact config is invalid."""
    return resolve_config(
        M,
        N,
        K,
        config=config,
        device=device,
        strict=False,
    )
