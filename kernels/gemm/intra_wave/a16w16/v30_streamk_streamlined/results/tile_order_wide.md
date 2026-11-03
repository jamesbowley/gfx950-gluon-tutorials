# v9 against v9full on the 55 wide points

**The kernel now uses v9full** (WORKLOG entry 13): in these tables "v9" is the previous order and "v9full"
is the current one.

`/tmp/v30order/wide_sweep.py`: timing_last_prefetch's 55 wide points (11 per tile, all five tiles), v9 (the kernel)
against `v30_v9full` (`/tmp/v30order/make_variants.py`: XCD blocks over `total_full_tiles`, Stream-K tiles = the
last S tile ids). timing_order's method, GPU 1, two runs of 9 rounds x 5 replays x 100 launches over 1 GB of
rotating copies, median. K % 64 != 0 means rows that are not 128-byte aligned (contiguous inputs).

- Geomean v9full / v9: 0.9986 and 0.9990 over all 55; 0.9977 and 0.9981 over the 40 with aligned rows.
- Gate (switch if the geomean is at most 1.0 and no aligned-row point regresses by more than 2% in both runs):
  **fails** on 3 points: 128x256x64 two_tile 2 trips K=8192 (1.026, 1.026), 128x256x64 spread K=16384 (1.034, 1.035), 64x64x64 two_tile 2 trips K=8192 (1.063, 1.065).
  The kernel kept v9's order at the time; v9full was adopted later for simplicity (entry 13).

| point | shape | plan | S | K % 64 | v9 us (run 1) | v9full / v9 (run 1, 2) | spread run 1 (v9/v9full) |
|---|---|---|---|---|---|---|---|
| 256x256x64 capped S=16 | 4352x4096x4096 | one_tile/rs | 16 | 0 | 113.73 | 0.985, 0.983 | 1.4%/0.6% |
| 256x256x64 rs S=16 ragged bias | 4320x4064x4080 | one_tile/rs | 16 | 48 | 143.17 | 1.035, 1.032 | 0.3%/0.3% |
| 256x256x64 owner S=16 3 waves | 12544x4096x4096 | one_tile/owner | 16 | 0 | 296.24 | 0.996, 0.994 | 0.7%/0.4% |
| 256x256x64 l2 S=16 K=8192 | 4352x4096x8192 | one_tile/rs | 16 | 0 | 213.76 | 0.999, 1.000 | 0.6%/2.6% |
| 256x256x64 two_tile | 8448x4096x4096 | two_tile/owner | 272 | 0 | 206.12 | 0.990, 0.988 | 2.0%/1.3% |
| 256x256x64 two_tile bias | 8448x4096x4096 | two_tile/owner | 272 | 0 | 209.86 | 0.992, 0.994 | 2.1%/1.1% |
| 256x256x64 two_tile ragged | 8416x4064x3984 | two_tile/owner | 272 | 16 | 255.81 | 0.998, 1.000 | 1.0%/0.6% |
| 256x256x64 two_tile 2 trips K=8192 | 12544x4096x8192 | two_tile/owner | 272 | 0 | 559.84 | 0.981, 0.979 | 1.8%/0.9% |
| 256x256x64 spread | 4096x7168x4096 | spread/owner | 192 | 0 | 174.93 | 1.002, 0.996 | 0.8%/0.9% |
| 256x256x64 spread ragged bias | 4064x7136x4080 | spread/owner | 192 | 48 | 225.52 | 1.002, 1.012 | 0.5%/0.5% |
| 256x256x64 spread K=16384 | 4096x7168x16384 | spread/owner | 192 | 0 | 626.69 | 1.008, 1.009 | 1.2%/0.9% |
| 256x128x64 capped S=16 | 4352x2048x4096 | one_tile/rs | 16 | 0 | 70.69 | 1.005, 1.011 | 0.8%/1.9% |
| 256x128x64 rs S=16 ragged bias | 4320x2032x4080 | one_tile/rs | 16 | 48 | 102.20 | 1.068, 1.065 | 1.8%/0.2% |
| 256x128x64 owner S=16 3 waves | 12544x2048x4096 | one_tile/owner | 16 | 0 | 184.28 | 0.993, 0.995 | 1.6%/0.7% |
| 256x128x64 l2 S=16 K=8192 | 4352x2048x8192 | one_tile/rs | 16 | 0 | 135.49 | 1.016, 1.020 | 2.1%/0.8% |
| 256x128x64 two_tile | 8448x2048x4096 | two_tile/owner | 272 | 0 | 131.89 | 0.959, 0.970 | 2.3%/1.3% |
| 256x128x64 two_tile bias | 8448x2048x4096 | two_tile/owner | 272 | 0 | 132.99 | 0.964, 0.969 | 1.5%/1.1% |
| 256x128x64 two_tile ragged | 8416x2032x3984 | two_tile/owner | 272 | 16 | 183.18 | 0.976, 0.976 | 0.4%/0.5% |
| 256x128x64 two_tile 2 trips K=8192 | 12544x2048x8192 | two_tile/owner | 272 | 0 | 344.38 | 0.987, 0.990 | 1.4%/0.4% |
| 256x128x64 spread | 4096x3584x4096 | spread/owner | 192 | 0 | 108.38 | 1.011, 1.015 | 1.8%/1.6% |
| 256x128x64 spread ragged bias | 4064x3568x4080 | spread/owner | 192 | 48 | 145.28 | 1.037, 1.036 | 0.5%/0.4% |
| 256x128x64 spread K=16384 | 4096x3584x16384 | spread/owner | 192 | 0 | 387.48 | 1.019, 1.024 | 0.5%/0.5% |
| 128x256x64 capped S=16 | 2176x4096x4096 | one_tile/rs | 16 | 0 | 69.26 | 1.009, 1.009 | 3.0%/1.6% |
| 128x256x64 rs S=16 ragged bias | 2160x4064x4080 | one_tile/rs | 16 | 48 | 99.86 | 1.029, 1.029 | 0.5%/0.2% |
| 128x256x64 owner S=16 3 waves | 6272x4096x4096 | one_tile/owner | 16 | 0 | 174.90 | 0.997, 1.000 | 0.9%/1.2% |
| 128x256x64 l2 S=16 K=8192 | 2176x4096x8192 | one_tile/rs | 16 | 0 | 132.70 | 1.014, 1.011 | 1.2%/0.6% |
| 128x256x64 two_tile | 4224x4096x4096 | two_tile/owner | 272 | 0 | 128.41 | 0.959, 0.961 | 0.9%/2.1% |
| 128x256x64 two_tile bias | 4224x4096x4096 | two_tile/owner | 272 | 0 | 131.64 | 0.953, 0.956 | 1.4%/1.1% |
| 128x256x64 two_tile ragged | 4208x4064x3984 | two_tile/owner | 272 | 16 | 176.71 | 0.908, 0.908 | 1.2%/0.5% |
| 128x256x64 two_tile 2 trips K=8192 | 6272x4096x8192 | two_tile/owner | 272 | 0 | 327.04 | 1.026, 1.026 | 1.2%/1.7% |
| 128x256x64 spread | 2048x7168x4096 | spread/owner | 192 | 0 | 108.39 | 0.971, 0.975 | 2.0%/1.5% |
| 128x256x64 spread ragged bias | 2032x7136x4080 | spread/owner | 192 | 48 | 149.50 | 0.931, 0.932 | 0.4%/0.3% |
| 128x256x64 spread K=16384 | 2048x7168x16384 | spread/owner | 192 | 0 | 391.01 | 1.034, 1.035 | 0.7%/0.3% |
| 128x128x128 capped S=16 | 2176x2048x4096 | one_tile/rs | 16 | 0 | 44.66 | 1.008, 1.004 | 0.8%/4.6% |
| 128x128x128 rs S=16 ragged bias | 2160x2032x4080 | one_tile/rs | 16 | 48 | 59.38 | 1.034, 1.034 | 0.7%/0.6% |
| 128x128x128 owner S=16 3 waves | 6272x2048x4096 | one_tile/owner | 16 | 0 | 109.47 | 0.999, 0.991 | 1.1%/1.7% |
| 128x128x128 l2 S=16 K=8192 | 2176x2048x8192 | one_tile/rs | 16 | 0 | 80.94 | 1.007, 1.007 | 1.0%/2.5% |
| 128x128x128 two_tile | 4224x2048x4096 | two_tile/owner | 272 | 0 | 79.80 | 0.986, 0.984 | 0.7%/0.9% |
| 128x128x128 two_tile bias | 4224x2048x4096 | two_tile/owner | 272 | 0 | 81.39 | 0.988, 0.988 | 1.1%/0.7% |
| 128x128x128 two_tile ragged | 4208x2032x3984 | two_tile/owner | 272 | 16 | 96.90 | 0.996, 0.995 | 0.7%/0.4% |
| 128x128x128 two_tile 2 trips K=8192 | 6272x2048x8192 | two_tile/owner | 272 | 0 | 201.09 | 0.992, 0.986 | 0.5%/0.5% |
| 128x128x128 spread | 2048x3584x4096 | spread/owner | 192 | 0 | 67.30 | 0.993, 0.994 | 0.7%/1.1% |
| 128x128x128 spread ragged bias | 2032x3568x4080 | spread/owner | 192 | 48 | 85.50 | 0.995, 0.995 | 0.6%/0.3% |
| 128x128x128 spread K=16384 | 2048x3584x16384 | spread/owner | 192 | 0 | 241.57 | 1.013, 1.008 | 1.3%/1.0% |
| 64x64x64 capped S=16 | 1088x1024x4096 | one_tile/rs | 16 | 0 | 30.30 | 1.015, 1.018 | 1.0%/0.9% |
| 64x64x64 rs S=16 ragged bias | 1080x1016x4080 | one_tile/rs | 16 | 48 | 36.27 | 1.030, 1.032 | 0.7%/0.3% |
| 64x64x64 owner S=16 3 waves | 3136x1024x4096 | one_tile/owner | 16 | 0 | 70.39 | 1.003, 1.001 | 0.4%/0.9% |
| 64x64x64 l2 S=16 K=8192 | 1088x1024x8192 | one_tile/rs | 16 | 0 | 54.37 | 1.017, 1.014 | 0.7%/0.7% |
| 64x64x64 two_tile | 2112x1024x4096 | two_tile/owner | 272 | 0 | 51.66 | 0.995, 0.993 | 0.7%/0.7% |
| 64x64x64 two_tile bias | 2112x1024x4096 | two_tile/owner | 272 | 0 | 53.38 | 0.994, 0.994 | 0.6%/0.4% |
| 64x64x64 two_tile ragged | 2104x1016x3984 | two_tile/owner | 272 | 16 | 58.88 | 0.989, 0.990 | 0.5%/0.6% |
| 64x64x64 two_tile 2 trips K=8192 | 3136x1024x8192 | two_tile/owner | 272 | 0 | 120.93 | 1.063, 1.065 | 0.3%/1.4% |
| 64x64x64 spread | 1024x1792x4096 | spread/owner | 192 | 0 | 42.61 | 0.987, 0.990 | 0.8%/0.5% |
| 64x64x64 spread ragged bias | 1016x1784x4080 | spread/owner | 192 | 48 | 48.17 | 0.996, 0.999 | 0.3%/0.2% |
| 64x64x64 spread K=16384 | 1024x1792x16384 | spread/owner | 192 | 0 | 145.29 | 0.988, 0.987 | 0.4%/2.0% |

## Footprint model

Per XCD, the distinct A tile-rows plus B tile-columns among the tiles it runs at once (see
[tile_order.md](tile_order.md)).

| point | tile | tiles (m x n) | G | S | v9 persistent mean / worst | v9full persistent mean / worst | v9 Stream-K worst | v9full Stream-K worst |
|---|---|---|---|---|---|---|---|---|
| 256x256x64 capped S=16 | 256x256x64 | 17x16 | 4 | 16 | 14.8 / 21.0 | 12.0 / 12.0 | 3 | 3 |
| 256x256x64 rs S=16 ragged bias | 256x256x64 | 17x16 | 4 | 16 | 14.8 / 21.0 | 12.0 / 12.0 | 3 | 3 |
| 256x256x64 owner S=16 3 waves | 256x256x64 | 49x16 | 4 | 16 | 14.3 / 18.0 | 12.0 / 12.0 | 3 | 3 |
| 256x256x64 l2 S=16 K=8192 | 256x256x64 | 17x16 | 4 | 16 | 14.8 / 21.0 | 12.0 / 12.0 | 3 | 3 |
| 256x256x64 two_tile | 256x256x64 | 33x16 | 4 | 272 | 12.5 / 13.0 | 12.0 / 12.0 | 21 | 21 |
| 256x256x64 two_tile bias | 256x256x64 | 33x16 | 4 | 272 | 12.5 / 13.0 | 12.0 / 12.0 | 21 | 21 |
| 256x256x64 two_tile ragged | 256x256x64 | 33x16 | 4 | 272 | 12.5 / 13.0 | 12.0 / 12.0 | 21 | 21 |
| 256x256x64 two_tile 2 trips K=8192 | 256x256x64 | 49x16 | 4 | 272 | 14.1 / 16.5 | 12.0 / 12.0 | 21 | 21 |
| 256x256x64 spread | 256x256x64 | 16x28 | 4 | 192 | 12.0 / 12.0 | 12.5 / 16.0 | 10 | 14 |
| 256x256x64 spread ragged bias | 256x256x64 | 16x28 | 4 | 192 | 12.0 / 12.0 | 12.5 / 16.0 | 10 | 14 |
| 256x256x64 spread K=16384 | 256x256x64 | 16x28 | 4 | 192 | 12.0 / 12.0 | 12.5 / 16.0 | 10 | 14 |
| 256x128x64 capped S=16 | 256x128x64 | 17x16 | 4 | 16 | 14.8 / 21.0 | 12.0 / 12.0 | 3 | 3 |
| 256x128x64 rs S=16 ragged bias | 256x128x64 | 17x16 | 4 | 16 | 14.8 / 21.0 | 12.0 / 12.0 | 3 | 3 |
| 256x128x64 owner S=16 3 waves | 256x128x64 | 49x16 | 4 | 16 | 14.3 / 18.0 | 12.0 / 12.0 | 3 | 3 |
| 256x128x64 l2 S=16 K=8192 | 256x128x64 | 17x16 | 4 | 16 | 14.8 / 21.0 | 12.0 / 12.0 | 3 | 3 |
| 256x128x64 two_tile | 256x128x64 | 33x16 | 4 | 272 | 12.5 / 13.0 | 12.0 / 12.0 | 21 | 21 |
| 256x128x64 two_tile bias | 256x128x64 | 33x16 | 4 | 272 | 12.5 / 13.0 | 12.0 / 12.0 | 21 | 21 |
| 256x128x64 two_tile ragged | 256x128x64 | 33x16 | 4 | 272 | 12.5 / 13.0 | 12.0 / 12.0 | 21 | 21 |
| 256x128x64 two_tile 2 trips K=8192 | 256x128x64 | 49x16 | 4 | 272 | 14.1 / 16.5 | 12.0 / 12.0 | 21 | 21 |
| 256x128x64 spread | 256x128x64 | 16x28 | 4 | 192 | 12.0 / 12.0 | 12.5 / 16.0 | 10 | 14 |
| 256x128x64 spread ragged bias | 256x128x64 | 16x28 | 4 | 192 | 12.0 / 12.0 | 12.5 / 16.0 | 10 | 14 |
| 256x128x64 spread K=16384 | 256x128x64 | 16x28 | 4 | 192 | 12.0 / 12.0 | 12.5 / 16.0 | 10 | 14 |
| 128x256x64 capped S=16 | 128x256x64 | 17x16 | 8 | 16 | 15.0 / 25.0 | 12.0 / 12.0 | 3 | 3 |
| 128x256x64 rs S=16 ragged bias | 128x256x64 | 17x16 | 8 | 16 | 15.0 / 25.0 | 12.0 / 12.0 | 3 | 3 |
| 128x256x64 owner S=16 3 waves | 128x256x64 | 49x16 | 8 | 16 | 14.4 / 22.3 | 12.0 / 12.0 | 3 | 3 |
| 128x256x64 l2 S=16 K=8192 | 128x256x64 | 17x16 | 8 | 16 | 15.0 / 25.0 | 12.0 / 12.0 | 3 | 3 |
| 128x256x64 two_tile | 128x256x64 | 33x16 | 8 | 272 | 12.8 / 13.0 | 12.0 / 12.0 | 25 | 25 |
| 128x256x64 two_tile bias | 128x256x64 | 33x16 | 8 | 272 | 12.8 / 13.0 | 12.0 / 12.0 | 25 | 25 |
| 128x256x64 two_tile ragged | 128x256x64 | 33x16 | 8 | 272 | 12.8 / 13.0 | 12.0 / 12.0 | 25 | 25 |
| 128x256x64 two_tile 2 trips K=8192 | 128x256x64 | 49x16 | 8 | 272 | 14.1 / 21.0 | 12.0 / 12.0 | 25 | 25 |
| 128x256x64 spread | 128x256x64 | 16x28 | 8 | 192 | 12.0 / 12.0 | 12.0 / 12.0 | 11 | 11 |
| 128x256x64 spread ragged bias | 128x256x64 | 16x28 | 8 | 192 | 12.0 / 12.0 | 12.0 / 12.0 | 11 | 11 |
| 128x256x64 spread K=16384 | 128x256x64 | 16x28 | 8 | 192 | 12.0 / 12.0 | 12.0 / 12.0 | 11 | 11 |
| 128x128x128 capped S=16 | 128x128x128 | 17x16 | 4 | 16 | 14.8 / 21.0 | 12.0 / 12.0 | 3 | 3 |
| 128x128x128 rs S=16 ragged bias | 128x128x128 | 17x16 | 4 | 16 | 14.8 / 21.0 | 12.0 / 12.0 | 3 | 3 |
| 128x128x128 owner S=16 3 waves | 128x128x128 | 49x16 | 4 | 16 | 14.3 / 18.0 | 12.0 / 12.0 | 3 | 3 |
| 128x128x128 l2 S=16 K=8192 | 128x128x128 | 17x16 | 4 | 16 | 14.8 / 21.0 | 12.0 / 12.0 | 3 | 3 |
| 128x128x128 two_tile | 128x128x128 | 33x16 | 4 | 272 | 12.5 / 13.0 | 12.0 / 12.0 | 21 | 21 |
| 128x128x128 two_tile bias | 128x128x128 | 33x16 | 4 | 272 | 12.5 / 13.0 | 12.0 / 12.0 | 21 | 21 |
| 128x128x128 two_tile ragged | 128x128x128 | 33x16 | 4 | 272 | 12.5 / 13.0 | 12.0 / 12.0 | 21 | 21 |
| 128x128x128 two_tile 2 trips K=8192 | 128x128x128 | 49x16 | 4 | 272 | 14.1 / 16.5 | 12.0 / 12.0 | 21 | 21 |
| 128x128x128 spread | 128x128x128 | 16x28 | 4 | 192 | 12.0 / 12.0 | 12.5 / 16.0 | 10 | 14 |
| 128x128x128 spread ragged bias | 128x128x128 | 16x28 | 4 | 192 | 12.0 / 12.0 | 12.5 / 16.0 | 10 | 14 |
| 128x128x128 spread K=16384 | 128x128x128 | 16x28 | 4 | 192 | 12.0 / 12.0 | 12.5 / 16.0 | 10 | 14 |
| 64x64x64 capped S=16 | 64x64x64 | 17x16 | 4 | 16 | 14.8 / 21.0 | 12.0 / 12.0 | 3 | 3 |
| 64x64x64 rs S=16 ragged bias | 64x64x64 | 17x16 | 4 | 16 | 14.8 / 21.0 | 12.0 / 12.0 | 3 | 3 |
| 64x64x64 owner S=16 3 waves | 64x64x64 | 49x16 | 4 | 16 | 14.3 / 18.0 | 12.0 / 12.0 | 3 | 3 |
| 64x64x64 l2 S=16 K=8192 | 64x64x64 | 17x16 | 4 | 16 | 14.8 / 21.0 | 12.0 / 12.0 | 3 | 3 |
| 64x64x64 two_tile | 64x64x64 | 33x16 | 4 | 272 | 12.5 / 13.0 | 12.0 / 12.0 | 21 | 21 |
| 64x64x64 two_tile bias | 64x64x64 | 33x16 | 4 | 272 | 12.5 / 13.0 | 12.0 / 12.0 | 21 | 21 |
| 64x64x64 two_tile ragged | 64x64x64 | 33x16 | 4 | 272 | 12.5 / 13.0 | 12.0 / 12.0 | 21 | 21 |
| 64x64x64 two_tile 2 trips K=8192 | 64x64x64 | 49x16 | 4 | 272 | 14.1 / 16.5 | 12.0 / 12.0 | 21 | 21 |
| 64x64x64 spread | 64x64x64 | 16x28 | 4 | 192 | 12.0 / 12.0 | 12.5 / 16.0 | 10 | 14 |
| 64x64x64 spread ragged bias | 64x64x64 | 16x28 | 4 | 192 | 12.0 / 12.0 | 12.5 / 16.0 | 10 | 14 |
| 64x64x64 spread K=16384 | 64x64x64 | 16x28 | 4 | 192 | 12.0 / 12.0 | 12.5 / 16.0 | 10 | 14 |
