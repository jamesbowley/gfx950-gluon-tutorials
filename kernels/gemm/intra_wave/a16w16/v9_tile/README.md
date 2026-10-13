# v9_tile: what limits a tile, worked through by hand

v9_tile is [v9_beyond_hotloop](../v9_beyond_hotloop/README.md) with the tile size as a parameter
(`bench_ext.py --version 9t --tile 256x128`). This page works through, with numbers, why smaller tiles
lose MFMA efficiency on MI355X (gfx950), and what limits each tile. The measurements behind it are in
[WORKLOG.md](WORKLOG.md).

The short version: per k-step, a tile's compute grows with its area and its loads with its edges.
Each CU can only move data so fast, so once the tile is small enough the MFMAs wait for data. Two
limits in the CU's L1 (the TCP) set that speed, and the kernel's LDS buffering adds a third.

## 1. Hardware constants

| constant | value | where it comes from |
|---|---|---|
| CUs, XCDs, SIMDs | 256 CUs = 8 XCDs x 32; 4 SIMDs per CU | |
| `v_mfma_f32_16x16x32_bf16` | 16x16x32 = 8192 MACs = 16,384 FLOP in 16 cycles | [v8 README](../v8_sliceMN/README.md), [process_json.py](../../../../../scripts/process_json.py) |
| one `buffer_load_dwordx4` | 64 lanes x 16 B = **1 KB** per wave | |
| TCP processing rate | one 64 B line per clock = **64 B/clk**, so 1 KB takes 16 cycles | [v8 README §4.3](../v8_sliceMN/README.md); measured as 153 GB/s per CU on L2 hits in hbm_bandwidth_xcd sweep I |
| TCP capacity | **32 KB** in flight per CU | [v8 README §4.5](../v8_sliceMN/README.md), [memory_bandwidth_model.md](../../../../../docs/memory_bandwidth_model.md) |
| `ds_read_b128` | 1 KB per wave, one every 16 cycles per wave with 4 waves | [v3 README](../v3_lds/README.md) |
| L2 | 2 KB/clk per XCD; 38.8 TB/s measured (39.3 at 2.4 GHz theoretical) | [hbm_bandwidth_xcd](../../../../../experiments/hbm_bandwidth_xcd/README.md) |
| MALL (Infinity Cache), as the XCDs see it | 7.0-7.9 TB/s (fabric-limited), ~110 GB/s per CU | same |
| HBM | 6.2-6.4 TB/s measured (8 TB/s theoretical), ~60 GB/s per CU | same |
| per-XCD fabric link | ~1.35 TB/s | same |
| power cap, clock | 1400 W, at most 2.4 GHz; 1.7-2.15 GHz under a full-chip GEMM | WORKLOG §4 |

### Derived rates

- **MFMA, per SIMD:** 16,384 FLOP / 16 cycles = 1024 FLOP/clk.
- **MFMA, per CU:** 4 x 1024 = **4096 FLOP/clk**. At 2.4 GHz that is 9.83 TFLOPS; at 1.92 GHz, 7.86.
- **MFMA, chip:** 256 x 4096 = 1,048,576 FLOP/clk, so **1.05 TFLOPS per GHz**: 2.52 PFLOPS at 2.4 GHz.
- **TCP processing, per CU:** 64 B/clk x 2.4 GHz = **153.6 GB/s**. That is what one CU reads from L2
  hits, measured in sweep I of
  [hbm_bandwidth_xcd](../../../../../experiments/hbm_bandwidth_xcd/README.md) ("I. L2 bandwidth
  scaling"; data in `results/gpu6/I.csv`, rows `x0_*`): 152-154 GB/s per CU with a 64 KB footprint,
  for 1-16 CUs on one XCD and 4, 8 or 16 waves alike. Because the wave count does not matter, it is
  the L1's processing rate, not an L2 limit. Only at 32 CUs per XCD does the XCD's L2 peak (2 KB/clk)
  show, at 126-138 GB/s per CU with 4-8 waves. At 1.92 GHz the TCP rate is 122.9 GB/s.
- **TCP processing, chip:** 256 x 64 B/clk = 16 KB/clk = 8 XCDs x 2 KB/clk. The L2 is sized to match
  the CUs exactly, so L2 never delivers more than the L1s can process. 16 KB/clk x 2.4 GHz =
  39.3 TB/s.
- **TCP capacity, per CU:** 32 KB in flight. By Little's law, bandwidth = bytes in flight /
  round-trip latency `L`. With `L` = 550 ns that is 32,768 / 550 = **59.6 GB/s**; with 385 ns,
  85.1 GB/s.

`L` = 550 ns (cold, operands from HBM) and 385 ns (warm, operands in the MALL) are *inferred*: they
are what turns 32 KB in flight into the measured ~60 and ~85 GB/s per CU (WORKLOG §4, §9). They are
averages over L2 hits and misses. A miss holds its slot several times longer than a hit, so with 80%
L2 hits the misses still occupy most of the window.

## 2. Per k-step quantities for a tile

A k-step is one `BLOCK_K` slice of the K loop: the CU loads an A strip (BM x BK) and a B strip
(BK x BN) into LDS, and the MFMAs multiply them into the BM x BN accumulator. The 4 waves sit in a
2x2 grid, one per SIMD, each owning a (BM/2) x (BN/2) piece.

| quantity | formula | 256x256x64 |
|---|---|---|
| MFMAs per wave per k-step | `(BM/2 / 16) x (BN/2 / 16) x (BK / 32)` | 8 x 8 x 2 = 128 |
| MFMA cycles per k-step, `C_cyc` | MFMAs per wave x 16 (the 4 waves run in parallel) | 2048 |
| MFMA time per k-step, `C` | `C_cyc / clock` | 2048 / 1.71 GHz = 1198 ns |
| bytes loaded per k-step, `B` | `(BM + BN) x BK x 2` | 64 KB |
| `buffer_load_dwordx4` per k-step | `B / 1 KB` (per wave: a quarter of that) | 64 (16 per wave) |
| TCP processing time per k-step | requests x 16 cycles | 1024 cycles |
| `ds_read_b128` per wave per k-step | `(BM/2 + BN/2) x BK x 2 / 1 KB` | 32 |
| accumulator registers per lane | `BM x BN / 256` | 256 |
| LDS, 2 buffers + padding | `2 x B x (1 + 16/512)` | 132 KB |
| intensity, FLOP per byte loaded | `2 BM BN BK / B = BM x BN / (BM + BN)` | 128 |

## 3. The limits, one at a time

Each is a ceiling on MFMA efficiency (the fraction of time the MFMAs are busy). The tile gets the
lowest one.

1. **TCP processing rate.** The tile needs `B / C_cyc` bytes per clock (the clock cancels). The TCP
   gives 64:
   `eff <= 64 / (B / C_cyc)`.
2. **TCP capacity, the window.** Hiding the latency needs the next `L / C` k-steps of data in flight
   at all times, `(B/C) x L` bytes; the TCP holds `Q / B` k-steps:
   `eff <= (Q / B) / (L / C) = (Q / L) / (B / C) = C / ((B / Q) x L)`.
   The last form says the same thing a third way: a k-step's bytes pass through the 32 KB window in
   `B / Q` round trips, which must fit in `C`.
3. **Pipeline depth, 2 LDS buffers (the kernel's choice).** k-step j's loads cannot be issued
   before k-step j-2 has freed its buffer, so they lead their use by about `2C`, which must cover
   the round trip plus the issue time `T_issue = requests x 16 cycles / clock`:
   `eff <= 2C / (L + T_issue)` (approximately; from the
   simulation in the tile-vs-TCP canvas).
4. **Shared bandwidth** (L2, MALL, HBM, the XCD links): chip demand against chip supply; see §5.

## 4. Worked table per tile

Full wave (256 tiles), each tile at its measured clock, cold (`L` = 550 ns), `Q` = 32 KB:

| tile | MFMAs / wave | `C_cyc` | clock | `C` | `B` | requests (per wave) | processing | demand | 1. processing | `L/C` | `Q/B` | 2. window | 3. 2 buffers | predicted | measured loop eff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256x256x64 | 128 | 2048 | 1.71 | 1198 ns | 64 KB | 64 (16) | 1024 cyc | 32 B/clk | 100% | 0.46 | 0.50 | 100% | 100% | **100%** | 95% |
| 256x128x64 | 64 | 1024 | 1.92 | 533 ns | 48 KB | 48 (12) | 768 cyc | 48 B/clk | 100% | 1.03 | 0.67 | 65% | 100% | **65%** | 64% |
| 128x256x64 | 64 | 1024 | 1.88 | 545 ns | 48 KB | 48 (12) | 768 cyc | 48 B/clk | 100% | 1.01 | 0.67 | 66% | 100% | **66%** | 68% |
| 128x128x64 | 32 | 512 | 2.11 | 243 ns | 32 KB | 32 (8) | 512 cyc | 64 B/clk | 100% | 2.27 | 1.00 | 44% | 61% | **44%** | 45% |
| 128x128x128 | 64 | 1024 | 2.10 | 488 ns | 64 KB | 64 (16) | 1024 cyc | 64 B/clk | 100% | 1.13 | 0.50 | 44% | 94% | **44%** | 49% |
| 64x64x64 | 8 | 128 | 2.11 | 61 ns | 16 KB | 16 (4) | 256 cyc | 128 B/clk | 50% | 9.07 | 2.00 | 22% | 18% | **18-22%** | not built |

Reading it:

- **256x256x64** is just inside the window: it needs 0.46 k-steps (29 KB) in flight and the TCP holds
  0.50 (32 KB). Two round trips per k-step (64 KB through a 32 KB window: 2 x 550 = 1100 ns) fit in
  the 1198 ns of compute.
- **256x128x64** halves the compute but loads 3/4 of the bytes. 1.5 round trips (825 ns) per 533 ns
  k-step: 533 / 825 = 65%.
- **128x128x64** uses exactly the TCP's processing rate (64 B/clk) at full speed, and needs 2.27
  k-steps (73 KB) in flight against 1 (32 KB) held: 44%. It is also the one tile where 2 LDS buffers
  are too shallow (61%). With 32 KB the window binds first, but the buffers are why ATT showed
  `vmcnt` waits.
- **128x128x128** doubles both `C` and `B`. The window ceiling is unchanged (44%), but the buffer
  ceiling rises from 61% to 94%, which is why BK = 128 removed the `vmcnt` waits and gained a few
  percent.
- **64x64x64** asks for 128 B/clk, twice what the TCP can process: at most 50% even with zero latency
  and unlimited slots.

Warm (operands in the MALL, `L` = 385 ns, measured warm clocks):

| tile | clock | `C` | `L/C` | window ceiling | measured loop eff |
|---|---|---|---|---|---|
| 256x128x64 | 1.75 GHz | 585 ns | 0.66 | ~100% | 89% |
| 128x128x64 | 1.96 GHz | 261 ns | 1.47 | 68% | 66% |
| 128x128x128 | 2.06 GHz | 497 ns | 0.77 | 65% | 62% |

### The same thing as bandwidth per CU

| tile | consumes `B/C` (cold clock) | window delivers `Q/L` | TCP processes `64 B/clk x clock` |
|---|---|---|---|
| 256x256x64 | 55 GB/s | 59.6 GB/s | 109 GB/s |
| 256x128x64 | 92 GB/s | 59.6 GB/s | 123 GB/s |
| 128x128x64 | 135 GB/s | 59.6 GB/s | 135 GB/s |
| 64x64x64 | 270 GB/s | 59.6 GB/s | 135 GB/s |

GB/s and bytes per ns are the same unit.

### Tiles with a side of 1 to 32

These run `v9_tile_small` (`--tile 16x256x128`, `1x256x128`, `256x16x128`; WORKLOG §14). The small side
is padded to the 16-row MFMA, and W waves sit side by side along the other side. A trailing `p` pins the
kernel to one wave per SIMD. Two limits join the three above:

- **MFMA padding** caps the useful share at `BM / 16`.
- **The LDS read rate** caps a wave tile `wm x wn` at `wm wn / (16 (wm + wn))`: 80% for 16x64, 50% for
  16x16.

On 2048x4096x8192 (cold, 2.4 GHz, pinned):

| tile | W | tiles/CU | `B` | window | 2 buffers | predicted busy | measured busy | useful | B/ns per CU | waits on |
|---|---|---|---|---|---|---|---|---|---|---|
| 32x128x128 | 2 | 2 | 40 KB | 16% | 39% | 16% | 22% | 21% | 82 | TCP issue |
| 32x256x64 | 4 | 1 | 36 KB | 17% | 27% | 17% | 17% | 17% | 59 | TCP issue, some `vmcnt` |
| 16x256x128 | 4 | 1 | 68 KB | 9% | 21% | 9% | 9.6% | 9.6% | 62 | TCP issue |
| 16x64x128 | 1 | 4 | 20 KB | 8% | 20% | 8% | 11% | 11% | 82 | TCP issue |
| 16x16x128 | 1 | 4 | 8 KB | 5% | 7% | 5% | 8.3% | 8.3% | 102 | **`vmcnt`** |
| 16x16x256 | 1 | 4 | 16 KB | 5% | 11% | 5% | 9.1% | 9.0% | 111 | TCP issue |
| 1x256x128 | 4 | 1 | 64 KB | 10% | 22% | 10% | 14% | 0.9% | 87 | TCP issue |

- **Every small tile is load-bound at 8-22% MFMA busy on a compute-bound shape.** One 4-wave workgroup per
  CU lands on the window model at ~60 B/ns per CU. Several one-wave workgroups per CU move 72-111 B/ns,
  so they beat the cold model, and warm inputs barely raise them: for that structure the 60 is not the
  limit.
- **Two buffers are enough** for BK >= 128 except 16x16x128, which waits on `vmcnt`. BK = 256 removes the
  waits (+10%).
- **Below 16 rows** the MFMA runs as busy as at 16 rows while the useful work shrinks to BM / 16.
- **Memory-bound shapes** (M = BM): the tile count and the bytes in flight per CU (`2 x tiles/CU x B`)
  set the bandwidth, not the intensity. About 2 tiles per CU at BK = 128 reach HBM, about 6 TB/s.
- **Triton's own choices** (WORKLOG §14.9):
  - The 4x64 / 64x4 MFMA (8 cycles, a quarter of 16x16x32's rate) gains nothing over padding to 16.
  - `v_dot2` below M = 4 is 3x slower.
  - Triton's generated code is 1.15-3.2x slower than v9_tile_small on these tiles; it stalls on
    `ds_bpermute` and LDS reads.
  - `waves_per_eu = 4 / W` pins one wave per SIMD exactly, without the LDS pad.
  - One-wave workgroups beat replicated warps for 16x16.

## 5. From one CU to the chip

### Throughput of a launch

A CU's rate is `min(R, I x Q/L)`, with `R` = 4096 FLOP/clk x clock and `I x Q/L` = intensity x
59.6 GB/s. A non-persistent launch of `T` tiles runs `ceil(T / 256)` waves, so on average
`T / ceil(T / 256)` CUs are busy. At a full wave the chip is also capped by power (~1590 TFLOPS for
256x256 on this GPU).

4096x2048x8192, bf16, back to back:

| tile | tiles | busy CUs | `I x Q/L` per CU | predicted | measured |
|---|---|---|---|---|---|
| 256x256 | 128 | 128 | 128 x 59.6 = 7.63 TFLOPS | 977 | 941 |
| 256x128 | 256 | 256 | 85.3 x 59.6 = 5.09 TFLOPS | 1302 | 1221 |
| 128x128 | 512 | 256 (2 waves) | 64 x 59.6 = 3.81 TFLOPS | 976 | 1027 |

At 4096x4096x8192, 256x256 would get 256 x 7.63 = 1953 TFLOPS from the load limit, but power caps it:
1593 measured.

### Is the per-CU limit above or below the shared bandwidth?

Below, while the L2 hit rate stays near 80%. For 256x128 on 4096x2048x8192 (cold):

| level | demand | supply |
|---|---|---|
| CUs from L2 (every request) | 256 x ~56 GB/s = 14.3 TB/s | 38.8 TB/s |
| beyond L2 (the ~20% misses) | 2.1M reads x 128 B / 112 us = 2.4 TB/s, ~0.3 TB/s per XCD | HBM 6.3, MALL 7.9 TB/s; 1.35 TB/s per XCD |

Spread over 256 CUs, the shared levels could feed each CU about 150 GB/s: L2 directly, and the misses
until 7.9 TB/s / (256 x 20%) = 154 GB/s. The window allows 60, so the window binds.

The L2 hit rate is also already at its geometric maximum. An XCD runs 32 tiles at once in a GM x
(32/GM) block, and only the block's outer strips come from beyond L2:
`1 - (GM x BM + (32/GM) x BN) / (32 x (BM + BN))`. That is 81.3% for 256x256 and 128x128 and 83.3%
for 256x128 at GM = 4; measured 79.5-81%, 80.4-81% and 82-83%.

### Why a memory-bound kernel can still saturate HBM

It faces the same window, but it needs less per CU. Streaming at the full 6.3 TB/s over 256 CUs is
6300 / 256 = **24.6 GB/s per CU**, well under the 59.6 a window allows, so HBM binds first. It needs at
least 6300 / 59.6 = **~106 CUs** pulling: [hbm_bandwidth_xcd](../../../../../experiments/hbm_bandwidth_xcd/README.md)
measured linear scaling up to about 130 CUs, then saturation at 6.0-6.3 TB/s. With 32 workgroups the
window binds: 32 x 59.6 = 1.9 TB/s (1.82 measured).

The small-tile GEMM is different. It needs ~92 GB/s per CU (256x128), more than any window
sustains. It can only ask for that much because 80% comes from L2.

## 6. Levers

| lever | effect on the inequality | in practice |
|---|---|---|
| bigger tile (higher `I`) | less `B` per FLOP | the main lever; limited by how many tiles the shape has |
| shorter `L` | raises `Q/L` | operands in the MALL: +30-45% per CU; L2 hit rate already at its geometric limit |
| lower clock | lengthens `C` | the power cap does this at full load; boosting does not help a ns-latency limit |
| more LDS buffers or larger BK | raises the pipeline-depth ceiling only | helped 128x128x64 (BK = 128); cannot raise the window |
| more bytes in flight (`Q`) | raises `Q/L` | fixed in hardware |
| split-K / stream-K | more busy CUs for the same tiles | trades a fixup for the idle CUs (WORKLOG §8) |
| GROUP_SIZE_M, small tiles | changes `L` through placement in L2 | the footprint rule leaves 20-26% on 16x256 and 16x64 (WORKLOG §14.6) |

The interactive canvas *Tile vs TCP window* (in the Cursor canvases directory) animates the slots,
the requests and the MFMA lane for any tile, clock, `L` and `Q`. It uses the same formulas.
