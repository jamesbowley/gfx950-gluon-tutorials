# Tile order: the ragged one-tile loss is the row stride and a row-shaped tail

**The kernel now uses v9full** (WORKLOG entry 13): in these tables "v9" is the previous order and "v9full"
is the current one.

Follow-up to [tile_order.md](tile_order.md), where v9full and split order were 3.5% slower than v9 on 4320x4064x4080 with
bias (one-tile rs, split 4, 256x256x64). Scripts in `/tmp/v30order` (`ragged_factors.py`, `ragged_stride*.py`), the
same timing method, GPU 1. Ratios are to v9. A and W are views of (M, LD) / (N, LD) buffers where a stride is given.

## 1. One factor at a time (4352x4096 base, rs split 4; two runs)

| point | v9 us | v9full / v9 | split / v9 |
|---|---|---|---|
| aligned (K=4096) | 113.4-113.7 | 0.998-1.000 | 0.997-1.002 |
| aligned, bias | 114.3-115.2 | 0.998-1.001 | 0.994-0.999 |
| ragged M (4320 / 4256) | 113.2-114.0 | 0.996-1.000 | 0.996-0.998 |
| ragged N (4064) | 113.9-114.8 | 0.998-1.000 | 0.997-0.999 |
| **ragged K (4080)** | 140.2-140.3 | **1.044-1.045** | **1.047-1.048** |
| ragged M, N and K | 140.7-141.6 | 1.032-1.036 | 1.031-1.036 |
| ragged M, N and K, bias | 143.0-143.5 | 1.032-1.036 | 1.027-1.031 |
| same, split 8 | 140.4 | 1.014-1.015 | 1.011-1.012 |

## 2. Ragged K against the row stride (4352x4096, rs split 4; two runs)

| point | v9 us | v9full / v9 | split / v9 |
|---|---|---|---|
| K=4096, stride 4096 | 109.8-109.9 | 1.001-1.004 | 0.998-1.002 |
| K=4096, stride 4112 (rows not 128 B aligned) | 139.0-139.4 | 1.048-1.049 | 1.048-1.049 |
| K=4080, stride 4080 (rows not 128 B aligned) | 139.0-139.2 | 1.046-1.047 | 1.048-1.049 |
| K=4080, stride 4096 (padded) | 109.8-109.9 | 1.000-1.003 | 1.002-1.007 |
| K=4032, stride 4032 (8064 B rows, K_RES 64) | 109.3-109.4 | 1.005-1.009 | 1.005-1.008 |
| K=4064, stride 4096 (padded, K_RES 32) | 109.7-109.8 | 1.000-1.005 | 1.003-1.006 |

Ragged K costs nothing with aligned rows. Rows that are not 128-byte aligned cost every order 27%, and the two orders
that put the Stream-K tiles last 5% more.

## 3. Other policies, and a persistent-only tail (stride 4112 against aligned; two runs)

| point | aligned: v9full / split | stride 4112: v9 us | stride 4112: v9full / v9 | stride 4112: split / v9 |
|---|---|---|---|---|
| dp 16x16 tiles (one wave) | 0.994-1.003 / 0.996-1.004 | 116.6 | 1.002-1.003 | 1.000-1.001 |
| dp 17x16 tiles (16-tile second wave) | 0.998-0.999 / 1.037 | 179.7-179.8 | 1.000-1.002 | **1.152** |
| rs S=16 | 1.001-1.002 / 1.000-1.002 | 139.0 | 1.048-1.049 | 1.048-1.049 |
| two-tile 33x16 | 0.991-0.992 / 0.989-0.990 | 260.4-260.7 | **0.959-0.960** | **0.959-0.960** |
| spread 16x28 | 1.001 / 1.001 | 207.3-207.5 | 1.005-1.006 | 1.002-1.004 |

dp has no Stream-K phase (v9full is v9's code there), so the 17x16 split-order loss comes from where the last partial
wave sits: one tile row, all on XCD 0.

## 4. The shape of the tail (one run)

`GROUP_SIZE_M` 17 makes the tile ids column-major, so the last 16 ids are a tile column instead of a row.

| point | last 16 tile ids | v9 us | v9full / v9 | split / v9 |
|---|---|---|---|---|
| rs S=16, G=4, stride 4112 | 1 row, 16 columns | 138.95 | 1.049 | 1.049 |
| rs S=16, G=17, stride 4112 | 16 rows, 1 column | 145.73 | 1.003 | 1.003 |
| rs S=16, G=4, aligned | 1 row, 16 columns | 110.18 | 1.001 | 0.998 |
| rs S=16, G=17, aligned | 16 rows, 1 column | 111.62 | 1.001 | 0.999 |
| rs S=16, 4096x4352 (16x17 tiles), G=4, stride 4112 | 4 rows, 4 columns | 138.07 | 1.004 | 1.003 |
| dp 17x16, G=4, stride 4112 | 1 row (XCD 0) | 179.30 | 1.000 | 1.154 |
| dp 17x16, G=1, stride 4112 | 1 row (XCD 0) | 185.79 | 1.000 | 1.154 |
| dp 17x16, G=17, stride 4112 | 1 column (XCD 0) | 186.63 | 1.000 | 0.991 |
| dp 17x16, G=17, aligned | 1 column (XCD 0) | 147.60 | 1.001 | 0.999 |

Only a tail that is one tile row is slow, and only with unaligned rows: its tiles share one A strip and between them
read every B strip. A column tail (one shared B strip, every A strip) and a 4x4 tail are not. A and W have the same
memory layout here, so the A / B asymmetry is unexplained; it would need cache counters.
