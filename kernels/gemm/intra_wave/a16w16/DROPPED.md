# Tried and dropped: stream-K ideas not to try again (without new evidence)

Everything the stream-K versions (v14-v29) tried and then removed, with the reason, the evidence and what would justify
reopening it. Open items are in [TODO.md](TODO.md); the history is in [STREAMK.md](STREAMK.md) and the per-version
work logs. v29 ([v29_streamk_compact](v29_streamk_compact/WORKLOG.md)) keeps only what remains: data-parallel,
one-tile with the owner or reduce-scatter ending, two-tile and spread, ragged M / N / K, small tiles, the capped rule,
`tile="auto"` and the l2 exchange. v28 still has every option below as code.

## Dropped for performance

**Splits and placement**
- **Forward order on an end-to-end split** (v15, v16, forward two-tile).
  - Programs start at their own K offsets and lose L2 reuse: 8% hits against 76% with aligned cuts on 3840x4096x8192.
  - Reversed order (v19) was 2-55% faster on every shape.
  - Evidence: [STREAMK.md, v19](STREAMK.md), [v19 WORKLOG](v19_streamk_two_tile_reversed/WORKLOG.md).
- **The uncapped one-tile split** (every program, about 256 / S per tile, v15-v20 defaults).
  - It over-splits, 64 ways at S = 4.
  - Capping it cut the owner ending's kernel time by 39% at S = 4, 10% at S = 16 and 4% at S = 38.
  - Evidence: [STREAMK.md, "Capping the split"](STREAMK.md).
- **Idle stream-K programs left on whole XCDs** (`STREAMK_BALANCE_XCDS=0`).
  - Spreading them over the XCDs was worth 11-15%.
  - Evidence: [STREAMK.md, "Capping the split"](STREAMK.md).
- **Chunk-major placement** (`STREAMK_PLACEMENT=chunk`, v18, v24).
  - Neutral: within ±3% of tile-major in v18, 0.99-1.01 in sweep F. Under v9 order tile-major already keeps the
    tiles that share a B column on one XCD.
  - Evidence: [STREAMK.md, v18 and v24](STREAMK.md), [v18 WORKLOG](v18_streamk_chunk_major/WORKLOG.md).
  - Reopen if: a placement study finds L2 reuse that tile-major misses.
- **A fixed owner on chunk 0 under chunk-major**: 2-8% slower than a rotating owner (+8.3% on 4352x4352x8192), as
  every owner sat on the same XCDs' links. Owners must be spread over the XCDs whenever programs are grouped by chunk
  ([STREAMK.md, v18](STREAMK.md)).
- **The one-tile uneven split for S > P / 2** (the `all` kind, `P % S` tiles on 2 programs).
  - Correct on all 161 points, but a median 1.24x behind the best of dp, pdp, spread, cap and two-tile; ahead by
    more than 2% on 1.
  - Evidence: [v27 WORKLOG §17](v27_streamk_tile/WORKLOG.md).
- **The split tile order** (`PERSISTENT_TILE_ORDER=split`).
  - v13 itself was 2-5% slower in it, and 3840x4096x8192 got worse in v15 (812 against 915 TFLOPS in v9 order).
  - Evidence: [v15 WORKLOG §1](v15_streamk_onetile/WORKLOG.md).

**Host rules**
- **`auto`** (`streamk_policy`, v20 entry 2): its two-tile thresholds were set against the uncapped split. Capped is
  the default since v20 entry 7.
  - Held-out grid: `auto` within 1.5% of the best on 24 of 32 shapes, geomean -1.56%.
  - Real-model shapes: within 1.5% on 457 of 630, geomean -4.50%.
  - With the split capped, one-tile ties or beats two-tile on most shapes.
  - Evidence: [v20 WORKLOG §7](v20_streamk_reduce_scatter/WORKLOG.md), [STREAMK.md, "Capping the split"](STREAMK.md).
- **`split_rule`** (the per-cell table, v20 entry 9): it beat capped on off-grid shapes timed cold but lost on the
  real-model shapes timed back to back. A small rule fitted to sweep F was no better than capped once cross-validated
  ([STREAMK.md, v24](STREAMK.md)).
  - Reopen if: a re-fit beats capped on held-out real-model shapes (v27 §15-17 is that line of work).

**Endings**
- **All readers in reduce-scatter and v20's row / column C store** (`STREAMK_RS_READERS=all`, no band store).
  - v23's power-of-two readers and band store were 0.5-4.4 us (2-8%) faster on every cell.
  - The band store cut the C store from up to 3.6 us to 0.4-1.0 us; power-of-two readers saved 1.6-4.4 us where n
    is not a power of two.
  - Evidence: [v23 WORKLOG](v23_streamk_rs_band/WORKLOG.md), [STREAMK.md, v23](STREAMK.md).
- **A separate zeroing launch before the atomic ending** (v22 `atomic`): a median 3.2 us slower than zeroing inside
  the GEMM (`atomic_iz`) ([v22 WORKLOG §2](v22_streamk_atomic_c/WORKLOG.md)). The atomic ending itself is now dropped
  too (below).
- **fp32 atomics into C**: slower than a store plus a read
  ([experiments/streamk_atomics](../../../../experiments/streamk_atomics/README.md)).
- **v27's per-quadrant owner ending** (one quadrant's peers read at a time, 4 loads per wait at 128x128): v28's batch
  of 16 loads per wave in a compile-time peer loop ("bs") is faster on every tile timed.
  - One-tile owner: 0.953-0.973 of v27's time at 128x128 and 256x128.
  - Same session against hipBLASLt 1.3.2: within 0.8% on 2 programs per tile.
  - Evidence: [v28 WORKLOG §8](v28_streamk_owner/WORKLOG.md).
- **v28's other owner variants** ([v28 WORKLOG §4, §5, §7, §9](v28_streamk_owner/WORKLOG.md)):
  - `p`, `atomic_poll` instead of the polled atomic. Only helps with 16+ peers (0.87-0.96), which capped never
    picks. Level to 3% slower at 2-4 peers, and failed the 256x256 regression guard (worst 1.034). Reopen only gated
    on n >= 16.
  - `w`, waiting for each peer just before its first batch. Neutral to worse, and 34 spills at 256x256 on two-tile
    and spread (up to 1.39x slower).
  - `f`, a register prefetch in the run-time peer loop. Level with b at best, up to 2.6% slower. A 128x128 peer
    already fills a CU's ~48 KB of loads in flight, so the drain between peers is not where the time goes.
  - The inline-asm tie (`together2` / `together4`) forcing a batch's loads to issue together: not needed once the
    loads are written before the adds.
  - Summing all four quadrants before storing C: spilled at 256x256 (four 64-VGPR sums beside the 256-AGPR
    accumulator). Storing each batch's C before the next batch's reads removed every spill.
- **Storing all four partial quadrants first in the 256x256 run-time owner loop** (v29's first form: end to end, and
  above 8 programs per tile): about 0.9% slower on two-tile in three sessions. v29 keeps v27's per-quadrant order there
  ([v29 WORKLOG §2](v29_streamk_compact/WORKLOG.md)).

**Code generation**
- **A masked tail prefetch** (`MASK_TAIL_PREFETCH`): about 12 VGPRs, which pushed the persistent kernel into spills;
  the last trip prefetches a clamped, valid tile instead ([v14 WORKLOG](v14_streamk/WORKLOG.md),
  [STREAMK_EXPLAINED](v14_streamk/STREAMK_EXPLAINED.md)).
- **Ragged K by shifting the whole K loop by `K_PAD - K`** (first in v9_ragged).
  - Misaligned every main-loop read with the 128-byte lines: 55% slower at K = 8176 and 8080 (287.9 and 283.8 us
    against 183.5).
  - The last, partial pair goes first instead.
  - Evidence: [v26 WORKLOG](v26_streamk_ragged_k/WORKLOG.md).
- **Edge handling on only one of M and N**: with only `EDGE_M` on, 512 VGPRs and 5-15 spills; both on, 492-496 and
  none ([v25 WORKLOG](v25_streamk_ragged_mn/WORKLOG.md)).

## Dropped for accuracy

Both endings below round C more than once. Neither was measured under AITER's `checkAllclose` (randn inputs, atol
and rtol 1e-2, failing above 5% of elements). By estimate both put several percent of a stream-K tile's elements
outside it, because their rounding error scales with the partial, not with the element. That would fail shapes with
less than one wave of tiles.

- **The atomic ending** (`STREAMK_FIXUP=atomic_iz`, v22-v28: every program adds its partial into C with packed bf16
  atomics).
  - C is rounded once per program, about `0.5 + 0.75 sqrt(n)` times one rounding's error (3.4 at n = 16, 6.2 at 64),
    and results are not bitwise reproducible.
  - Like for like it was 3.8% behind bf16-partial reduce-scatter in sweep F (shorter in 25 of 90 cells).
  - Evidence: [v22 WORKLOG](v22_streamk_atomic_c/WORKLOG.md), [v23 WORKLOG](v23_streamk_rs_band/WORKLOG.md),
    [STREAMK.md, v24](STREAMK.md).
  - Reopen if: a measured pass under AITER's `checkAllclose` on shapes below one wave, and a lead over fp32
    reduce-scatter.
- **bf16 and bf16-pair partials** (`STREAMK_PARTIALS`, v23-v28).
  - The largest single lever: 14% off reduce-scatter's tail and 7% off the owner's in sweep F, at about 1.6 times one
    rounding's error whatever n.
  - Under bench.py's `allclose` (atol 0.1, rtol 0.01) 13 and 12 configurations already failed, all 2-way splits.
  - The pair map is bitwise identical to plain bf16, only faster to read.
  - Evidence: [v23 WORKLOG](v23_streamk_rs_band/WORKLOG.md), [STREAMK.md, v24](STREAMK.md).
  - Reopen if: a measured pass under AITER's `checkAllclose`. Plain bf16 (no pair map) is the cheap form to bring
    back first.

## Dropped for code clarity

No measurement favoured any of these; they went in v29 to shorten the code.

- **The l2 exchange without the XCD-local split** (`STREAMK_XCD_LOCAL=0`), its placement replay (`tile_groups`,
  `first_straddling_tile`) and the `STREAMK_UNSAFE_PLACEMENT` negative control.
  - The balanced placement only works for l2 when no tile happens to straddle XCDs. The XCD-local split cannot
    straddle, and the host now asserts it.
  - l2 itself stays (rs only). It wins at K <= 1024 and loses 20-47% at S = 2-8 for K >= 4096 ([STREAMK.md,
    v24](STREAMK.md)).
- **`STREAMK_PERSISTENT_PROGRAMS`** (persistent dp on 512-1024 programs).
  - It matched v9_tile on the small quadrant tiles: 145 to 130 us at 128x128x64 on 4096³, against v9_tile's 129
    ([v27 WORKLOG](v27_streamk_tile/WORKLOG.md)).
  - Dropped from v29 for the time being; extending it to stream-K for skinny, memory-bound shapes is in
    [TODO.md](TODO.md).
- **P slot padding** (`STREAMK_P_PAD`): neither the placement nor the alignment of `P` mattered (median -0.0% fresh,
  +0.1% padded) ([v20 WORKLOG §12](v20_streamk_reduce_scatter/WORKLOG.md)).
- **The compact launch's off switch** (`STREAMK_COMPACT_GRID=0`): the compact launch measured level with the full
  grid (0.994-1.007 on re-timed points) and is always on in v29 ([v27 WORKLOG §13](v27_streamk_tile/WORKLOG.md)).
- **Experiment and debug tooling** (still in v28):
  - `GEMM_TILE_LIMIT`: a GEMM of the grid's first T tiles, for the S sweeps.
  - `STREAMK_RECORD_PLACEMENT`: every program's hardware placement. It found that workgroup p runs on XCD
    (p + r) % 8, with r varying between launches ([v27 WORKLOG §13](v27_streamk_tile/WORKLOG.md)).
  - `V9_TILE_GM`: a GROUP_SIZE_M override.
