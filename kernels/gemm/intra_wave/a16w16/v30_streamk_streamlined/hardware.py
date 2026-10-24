"""Runtime topology discovery and optional LDS diagnostics for v30."""

import functools

import torch
from aiter.ops.triton.utils.device_info import get_num_sms, get_num_xcds

from .selection import DeviceConfig, TileSpec, cdiv

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


def _gfx():
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    gfx = properties.gcnArchName.split(":", 1)[0]
    if not gfx:
        raise RuntimeError("the active device does not report a GCN architecture")
    return gfx


@functools.lru_cache(maxsize=None)
def streamk_topology():
    """Return the validated ``(num_programs, num_xcds)`` of the current device."""
    num_programs, num_xcds = int(get_num_sms()), int(get_num_xcds())
    if num_programs <= 0 or num_xcds <= 0 or num_programs % num_xcds:
        raise RuntimeError(
            f"invalid runtime topology: {num_programs} CUs, {num_xcds} XCDs"
        )
    return num_programs, num_xcds


@functools.lru_cache(maxsize=None)
def query_device_config():
    """Return AITER-compatible CU/XCD topology for the current device."""
    num_programs, num_xcds = streamk_topology()
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return DeviceConfig(
        gfx=_gfx(),
        num_programs=num_programs,
        num_xcds=num_xcds,
        lds_limit_bytes=int(properties.shared_memory_per_block),
    )


def estimate_tile_lds_bytes(tile, dtype):
    """Conservatively estimate this kernel's padded, double-buffered LDS.

    The layouts add 16 elements after each 512 logical elements.  A 9/8 margin
    covers the additional shared allocation observed in compiled v30 kernels;
    the compiler's resource check remains the final authority.
    """
    if not isinstance(tile, TileSpec):
        raise TypeError("tile must be a TileSpec")
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"v30 supports fp16 and bf16 inputs, got {dtype}")

    allocations = (
        tile.block_m * tile.block_k,
        tile.block_m * tile.block_k,
        tile.block_n * tile.block_k,
        tile.block_n * tile.block_k,
    )
    padded_elements = sum(
        elements + cdiv(elements, 512) * 16 for elements in allocations
    )
    padded_bytes = padded_elements * 2
    return cdiv(padded_bytes * 9, 8)


def validate_tile_lds(tile, dtype, device):
    """Diagnose whether the conservative estimate exceeds the device limit."""
    if not isinstance(device, DeviceConfig):
        raise TypeError("device must be a DeviceConfig")
    required = estimate_tile_lds_bytes(tile, dtype)
    if required > device.lds_limit_bytes:
        raise ValueError(
            f"tile {tile} is estimated to require {required} B of LDS, "
            f"more than {device.gfx}'s {device.lds_limit_bytes} B limit"
        )
    return required
