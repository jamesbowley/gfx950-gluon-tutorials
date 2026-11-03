# v30 tile orders: v9, v9full and split order

**The kernel now uses v9full** (WORKLOG entry 13): in these tables "v9" is the previous order and "v9full"
is the current one.

The 55-point sweep in [tile_order_wide.md](tile_order_wide.md) (WORKLOG entry 11) found v9full up to 6.5% slower on
three aligned points; it was adopted anyway, for simplicity (entry 13).

Timed with `/tmp/v30order/timing_order.py` (variants from `/tmp/v30order/make_variants.py`, model from
`/tmp/v30order/footprint.py`), GPU 1, two runs of 9 rounds x 5 replays x 100 launches over 1 GB of rotating
copies; median us per launch. Every order checked against torch before and after the graph warm-up.

- **v9**: the kernel's order. XCD blocks over `total_tiles`, Stream-K tiles at the top of each block.
- **v9full**: XCD blocks over `total_full_tiles`, Stream-K tiles = the last S tile ids.
- **split**: remapped program id + i P, Stream-K tiles = the last S tile ids.

| point | tile | shape | plan | S | v9 us (run 1, 2) | v9full / v9 (run 1, 2) | split / v9 (run 1, 2) | spread run 1 (v9/v9full/split) |
|---|---|---|---|---|---|---|---|---|
| rs S=8 K=8192 | 256x256x64 | 3072x5632x8192 | one_tile/rs | 8 | 202.23, 202.49 | 0.998, 0.996 | 0.996, 0.996 | 0.7%/0.5%/0.6% |
| rs S=16 K=8192 | 256x256x64 | 4352x4096x8192 | one_tile/rs | 16 | 211.24, 211.26 | 0.996, 0.995 | 0.993, 0.989 | 1.2%/0.7%/1.3% |
| rs S=32 K=8192 | 256x256x64 | 4096x4608x8192 | one_tile/rs | 32 | 220.45, 220.37 | 1.011, 1.006 | 1.010, 0.992 | 1.7%/3.2%/3.0% |
| capped S=16 K=4096 | 256x256x64 | 4352x4096x4096 | one_tile/rs | 16 | 113.87, 114.25 | 0.984, 0.980 | 0.979, 0.976 | 1.3%/1.5%/1.5% |
| capped 129x6 K=4096 | 256x256x64 | 33024x1536x4096 | one_tile/rs | 6 | 291.16, 291.95 | 0.984, 0.982 | 0.988, 0.988 | 0.8%/0.7%/1.3% |
| two_tile K=4096 | 256x256x64 | 8448x4096x4096 | two_tile/owner | 272 | 206.20, 206.89 | 0.987, 0.987 | 0.988, 0.988 | 1.0%/1.1%/1.2% |
| two_tile K=8192 | 256x256x64 | 8448x4096x8192 | two_tile/owner | 272 | 383.70, 384.34 | 0.984, 0.984 | 0.983, 0.983 | 1.3%/0.6%/0.5% |
| two_tile K=16384 | 256x256x64 | 8448x4096x16384 | two_tile/owner | 272 | 751.70, 752.19 | 0.988, 0.987 | 0.987, 0.987 | 0.9%/0.4%/0.7% |
| spread K=4096 | 256x256x64 | 4096x7168x4096 | spread/owner | 192 | 175.33, 175.32 | 1.001, 0.999 | 0.998, 0.999 | 1.1%/0.7%/1.4% |
| spread K=16384 | 256x256x64 | 4096x7168x16384 | spread/owner | 192 | 626.06, 627.10 | 1.015, 1.015 | 1.014, 1.013 | 0.5%/0.9%/0.7% |
| dp 1 wave | 256x256x64 | 4096x4096x8192 | dp/owner | 0 | 177.88, 177.80 | 0.997, 1.003 | 0.998, 1.002 | 1.4%/1.7%/1.4% |
| dp 4 waves | 256x256x64 | 8192x8192x8192 | dp/owner | 0 | 689.79, 690.56 | 1.001, 1.000 | 1.001, 1.001 | 0.7%/1.1%/1.2% |
| rs ragged bias | 256x256x64 | 4320x4064x4080 | one_tile/rs | 16 | 143.11, 142.85 | 1.035, 1.037 | 1.031, 1.034 | 0.6%/0.3%/0.3% |
| two_tile ragged bias | 256x256x64 | 8416x4064x3984 | two_tile/owner | 272 | 260.88, 261.14 | 1.005, 1.004 | 1.001, 0.999 | 0.6%/0.6%/0.9% |
| spread ragged bias | 256x256x64 | 4064x7136x4080 | spread/owner | 192 | 223.43, 223.65 | 0.999, 0.998 | 0.996, 0.997 | 0.3%/0.4%/0.7% |
| rs S=16 K=8192 | 128x128x128 | 2176x2048x8192 | one_tile/rs | 16 | 77.99, 78.07 | 0.991, 0.992 | 0.989, 0.991 | 1.1%/0.9%/0.8% |
| capped 129x6 K=4096 | 128x128x128 | 16512x768x4096 | one_tile/rs | 6 | 112.81, 113.19 | 0.965, 0.967 | 0.988, 0.987 | 0.8%/1.8%/0.5% |
| two_tile K=8192 | 128x128x128 | 4224x2048x8192 | two_tile/owner | 272 | 141.23, 141.31 | 0.981, 0.982 | 0.980, 0.980 | 1.4%/1.6%/1.1% |
| spread K=8192 | 128x128x128 | 2048x3584x8192 | spread/owner | 192 | 117.65, 117.93 | 1.000, 0.999 | 0.997, 0.997 | 1.1%/1.2%/1.1% |
| dp 1 wave | 128x128x128 | 2048x2048x8192 | dp/owner | 0 | 67.22, 67.31 | 0.998, 0.998 | 0.997, 0.997 | 0.8%/1.5%/0.8% |

## Footprint model

Per XCD, the distinct A tile-rows plus B tile-columns among the tiles it runs at once: its 32 programs'
tiles in one persistent trip (mean over XCDs and trips / mean over trips of the worst XCD), or its programs'
Stream-K tiles (worst XCD). G is the plan's `GROUP_SIZE_M`.

| point | tile | tiles (m x n) | G | S | v9 persistent mean / worst | v9full persistent mean / worst | split persistent mean / worst | v9 Stream-K worst | v9full Stream-K worst | split Stream-K worst |
|---|---|---|---|---|---|---|---|---|---|---|
| rs S=8 K=8192 | 256x256x64 | 12x22 | 4 | 8 | 13.8 / 17.0 | 13.0 / 16.0 | 13.0 / 16.0 | 2 | 2 | 2 |
| rs S=16 K=8192 | 256x256x64 | 17x16 | 4 | 16 | 14.8 / 21.0 | 12.0 / 12.0 | 12.0 / 12.0 | 3 | 3 | 3 |
| rs S=32 K=8192 | 256x256x64 | 16x18 | 4 | 32 | 12.0 / 12.0 | 13.5 / 16.0 | 13.5 / 16.0 | 5 | 5 | 5 |
| capped S=16 K=4096 | 256x256x64 | 17x16 | 4 | 16 | 14.8 / 21.0 | 12.0 / 12.0 | 12.0 / 12.0 | 3 | 3 | 3 |
| capped 129x6 K=4096 | 256x256x64 | 129x6 | 4 | 6 | 14.6 / 15.3 | 14.0 / 14.0 | 14.0 / 14.0 | 4 | 3 | 3 |
| two_tile K=4096 | 256x256x64 | 33x16 | 4 | 272 | 12.5 / 13.0 | 12.0 / 12.0 | 12.0 / 12.0 | 21 | 21 | 21 |
| two_tile K=8192 | 256x256x64 | 33x16 | 4 | 272 | 12.5 / 13.0 | 12.0 / 12.0 | 12.0 / 12.0 | 21 | 21 | 21 |
| two_tile K=16384 | 256x256x64 | 33x16 | 4 | 272 | 12.5 / 13.0 | 12.0 / 12.0 | 12.0 / 12.0 | 21 | 21 | 21 |
| spread K=4096 | 256x256x64 | 16x28 | 4 | 192 | 12.0 / 12.0 | 12.5 / 16.0 | 12.5 / 16.0 | 10 | 14 | 14 |
| spread K=16384 | 256x256x64 | 16x28 | 4 | 192 | 12.0 / 12.0 | 12.5 / 16.0 | 12.5 / 16.0 | 10 | 14 | 14 |
| dp 1 wave | 256x256x64 | 16x16 | 4 | 0 | 12.0 / 12.0 | 12.0 / 12.0 | 12.0 / 12.0 | - | - | - |
| dp 4 waves | 256x256x64 | 32x32 | 4 | 0 | 12.0 / 12.0 | 12.0 / 12.0 | 12.0 / 12.0 | - | - | - |
| rs ragged bias | 256x256x64 | 17x16 | 4 | 16 | 14.8 / 21.0 | 12.0 / 12.0 | 12.0 / 12.0 | 3 | 3 | 3 |
| two_tile ragged bias | 256x256x64 | 33x16 | 4 | 272 | 12.5 / 13.0 | 12.0 / 12.0 | 12.0 / 12.0 | 21 | 21 | 21 |
| spread ragged bias | 256x256x64 | 16x28 | 4 | 192 | 12.0 / 12.0 | 12.5 / 16.0 | 12.5 / 16.0 | 10 | 14 | 14 |
| rs S=16 K=8192 | 128x128x128 | 17x16 | 4 | 16 | 14.8 / 21.0 | 12.0 / 12.0 | 12.0 / 12.0 | 3 | 3 | 3 |
| capped 129x6 K=4096 | 128x128x128 | 129x6 | 4 | 6 | 14.6 / 15.3 | 14.0 / 14.0 | 14.0 / 14.0 | 4 | 3 | 3 |
| two_tile K=8192 | 128x128x128 | 33x16 | 4 | 272 | 12.5 / 13.0 | 12.0 / 12.0 | 12.0 / 12.0 | 21 | 21 | 21 |
| spread K=8192 | 128x128x128 | 16x28 | 4 | 192 | 12.0 / 12.0 | 12.5 / 16.0 | 12.5 / 16.0 | 10 | 14 | 14 |
| dp 1 wave | 128x128x128 | 16x16 | 4 | 0 | 12.0 / 12.0 | 12.0 / 12.0 | 12.0 / 12.0 | - | - | - |
