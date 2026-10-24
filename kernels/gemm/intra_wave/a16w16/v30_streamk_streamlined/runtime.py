"""Workspace management for the v30 Stream-K launcher.

Flags are zeroed once per (device, stream) and every launch hands them back
zeroed: owners clear their peers' locks, the last reduce-scatter arrival clears
the count, and L2 readers clear their own rows.  Generation words only
increment, so the flag layout is fixed for every configuration.

Buffers come from AITER's persistent pool so they outlive CUDA graph captures.
If the flags are first created while capturing, their zero fill is recorded
into the graph and runs on every replay; one eager launch on the capture
stream beforehand avoids that.
"""

import functools

import torch
from aiter.utility.graph_alloc import persistent_alloc

from .selection import QUADRANT_SIDES

MAX_TILE_ELEMS = max(QUADRANT_SIDES) ** 2


def _sync_words(num_programs, num_xcds):
    return 2 * num_programs + num_programs * (num_programs // num_xcds)


# Bounded and fixed-size to match AITER's FlyDSL split-K buffers
# (_get_preshuffle_split_buffers), with the same risk: past 128 (device,
# stream) keys an evicted buffer returns to the pool while a captured CUDA
# graph may still point at it.
@functools.lru_cache(maxsize=128)
def _flags(device, stream, num_programs, num_xcds):
    with persistent_alloc(device):
        return torch.zeros(
            _sync_words(num_programs, num_xcds) + num_programs,
            dtype=torch.int32,
            device=device,
        )


@functools.lru_cache(maxsize=128)
def _persistent_partials(device, stream, num_programs):
    with persistent_alloc(device):
        return torch.empty(
            num_programs * MAX_TILE_ELEMS, dtype=torch.float32, device=device
        )


def streamk_workspace(
    device,
    num_programs,
    num_xcds,
    block_m,
    block_n,
    streamk_programs,
    l2,
    persistent_partials=False,
):
    """Return ``(partials, locks, sync)`` views for one launch.

    ``sync`` is the reduce-scatter (count, generation) pairs followed by one
    programs-per-XCD L2 flag row per program slot (128-byte aligned rows);
    one owner lock per program follows it.  Partial slots are indexed by
    Stream-K program id, which spans the whole resident grid for the L2
    exchange.  Per-call partials need no zeroing: every slot is written before
    it is read.
    """
    device = torch.device(device)
    if device.index is None:
        device = torch.device(device.type, torch.cuda.current_device())
    stream = torch.cuda.current_stream(device)
    sync_words = _sync_words(num_programs, num_xcds)
    flags = _flags(device, stream, num_programs, num_xcds)
    if persistent_partials:
        partials = _persistent_partials(device, stream, num_programs)
        partials = partials[: num_programs * block_m * block_n]
    else:
        slots = num_programs if l2 else streamk_programs
        partials = torch.empty(
            slots * block_m * block_n, dtype=torch.float32, device=device
        )
    return (
        partials,
        flags[sync_words : sync_words + streamk_programs],
        flags[:sync_words],
    )
