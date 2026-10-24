# v30_streamk_streamlined

V30 is the quadrant-only production-shaped form of v29's Stream-K kernel. It
separates tile/split selection from execution and removes experimental
environment controls.

## Supported kernel

- gfx950 fp16/bf16 `C = A @ B (+ bias)`.
- `BLOCK_M`, `BLOCK_N`: 64, 128, or 256.
- `BLOCK_K`: 64 or 128.
- Four waves per program and one persistent program per queried CU.
- Persistent DP, uniform one-tile Stream-K, two-tile, and spread.
- Owner and reduce-scatter one-tile endings; WT and XCD-local L2 exchange.
- Every flag, count and generation is re-armed inside the launch, so every
  path, L2 included, is safe under CUDA graph replay.
- Ragged M, N, and K, bias, and compact no-full-tile grids.

Small tiles are intentionally not part of this initial version. A tile with
either side below 64 is rejected; v30 does not silently fall back to another
kernel.

## Scheduling policies

Let `P` be the resident program count, `T` the output tile count, and
`S = T % P` the tiles left after the last full wave. A K-pair covers two
`BLOCK_K` steps. For `P=8` and eight K-pairs per tile:

- **DP**, with `T=10` and `S=2`: all tiles use the persistent loop; programs
  0 and 1 process the two tiles after the first wave.
- **One-tile**, with `T=10`, `S=2`, and `split_n=4`: the first eight tiles use
  the persistent loop, then each tail tile is split across four programs, with
  two K-pairs per program.
- **Two-tile**, with `T=18` and `S=2`: the first eight tiles use the persistent
  loop, then the final `P+S=10` tiles' 80 K-pairs are distributed end to end
  across all eight programs.
- **Spread**, with `T=13` and `S=5`: the first eight tiles use the persistent
  loop, then the five tail tiles' 40 K-pairs are distributed end to end across
  all eight programs. Spread is valid only when `S > P/2`.

Two-tile and spread process their end-to-end ranges in reverse tile order.

## Input layout

The public launcher follows AITER's linear convention:

- A has logical shape `(M, K)` with `stride_ak == 1`.
- Weights have logical shape `(N, K)` with K contiguous.
- C is row-major.
- Leading strides may be padded but must be multiples of eight elements.

The launcher creates a zero-copy `(K,N)` transposed view of the weights for
the device kernel.

Negative strides and tile-local byte offsets outside signed 32-bit range are
rejected by the launcher.

## Explicit configurations

`gemm_a16w16` accepts the flat primitive-only dictionary used by AITER. Every
tunable value is explicit; the launcher derives only mechanical constants from
the shape and runtime topology:

```python
from v30_streamk_streamlined.matmul_kernel import gemm_a16w16

config = {
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "BLOCK_K": 128,
    "GROUP_SIZE_M": 4,
    "STREAMK_POLICY": "one_tile",
    "STREAMK_FIXUP": "rs",
    "SPLIT_N": 8,
    "STREAMK_EXCHANGE": "wt",
}
c = gemm_a16w16(a, weights, bias=bias, config=config)
```

The required policy is one of `dp`, `one_tile`, `two_tile`, or `spread`;
the required fixup is `owner` or `rs`. `SPLIT_N` is an integer for
`one_tile` and `None` otherwise. `GROUP_SIZE_M` and `STREAMK_EXCHANGE`
(`wt` or `l2`) are also required. `capped`, `auto`, missing tile dimensions,
and omitted splits are rejected on this runtime path.

The launcher obtains CU and XCD counts through AITER-compatible host helpers,
then derives `S * split_n`, compact grid size, reader count, pair partitioning,
and XCD placement. Tile-aligned uneven splits and a raw active-program override
are not supported. Two-tile and spread always use all resident programs.

Experimental auto-tile and capped rules remain isolated in `heuristics.py` for
offline config generation. They emit a complete dictionary before launch;
neither `selection.py` nor `matmul_kernel.py` calls them. `selection.py`
provides the same exact-config validation and mechanical calculations for
offline pruning; the launcher does not call its resolver.

## Ragged K

`K_PAD` is a virtual scheduling extent, not tensor padding. The final masked
physical pair runs first, followed by aligned full pairs from column zero.
This keeps every segment in two-step units and leaves the steady-state pair
loop unmasked.

## L2 exchange

With `STREAMK_EXCHANGE="l2"` every program of a split tile runs on one XCD,
and the partials and flags stay in that XCD's L2. There are no atomics and
no host-side epoch. Each reader has a private row with one flag per writer
in the tile; writers store 1 to their word in every row, and each reader
polls its row with one L1-missing load and zeroes it once all are set. Only
one wave of a program stores or resets flags, behind a workgroup barrier.
See [WORKLOG.md](WORKLOG.md) entry 5.

The partial workspace and flags are shared per device and tile size, so two
streams must not run v30 Stream-K launches on one device concurrently. That
applies to the WT path's counters too.

## Checks

From the `a16w16` directory:

```bash
source v29_streamk_compact/env.sh
python v30_streamk_streamlined/check_streamk.py --tile 64x64x64
bash v30_streamk_streamlined/run_checks.sh
```

`--graph` replays captured L2 launches (single-node, two-node, and mixed
with eager launches) over fresh inputs and poisoned partials. The
stale-count mode is a negative control and must time out; `--stale-flag`
presets the L2 flags and must give a wrong result (at 128x128x128).

`asm_check.py` counts the synchronization instructions of the WT and L2
builds, and `timing.py` times L2 against v29's epoch flags and WT in CUDA
graphs.
