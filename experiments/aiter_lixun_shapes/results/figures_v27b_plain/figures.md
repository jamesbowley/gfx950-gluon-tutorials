# Figures: streamk_wip (v20, capped) against the best of AITER's pick and hipBLASLt

Data: `details_v27b_merged_rot2048.csv` (cold: 2 GB rotating buffer): v27 with the revised auto tile rule (`1175b735`, timed 15:4x), the libraries, v9s and v26 from the 2026-10-15 13:29 sweep, HIP GPU 3, everything contiguous.

The baseline on each shape is the faster of AITER's pick and hipBLASLt (the torch call AITER's `torch` libtype makes), with that kernel's tile. AITER itself dispatches the winner of its offline tuning, so this is a stricter bar than what AITER runs today. "Ahead" and "behind" mean a gap of more than 1%.

## Figure 0: summary

![fig0](fig0_summary.png)

"Data-parallel only" is streamk_wip with `STREAMK_POLICY=dp`: the same persistent kernel without the stream-K split. v9 to data-parallel only is the persistent kernel's change; data-parallel only to streamk_wip is what stream-K adds.

| Caches | Shapes | Count | v9: faster | v9: within 1% | v9: slower | v9: median | new v9: faster | new v9: within 1% | new v9: slower | new v9: median | v26: faster | v26: within 1% | v26: slower | v26: median | data-parallel only: faster | data-parallel only: within 1% | data-parallel only: slower | data-parallel only: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cold | All shapes | 630 | 248 | 38 | 344 | -3.9% | 334 | 38 | 258 | +1.6% | 361 | 33 | 236 | +2.2% | 348 | 47 | 235 | +1.6% | 446 | 44 | 140 | +3.4% |
| Cold | Under 256 tiles (less than one wave) | 321 | 25 | 8 | 288 | -36.5% | 107 | 21 | 193 | -4.8% | 79 | 15 | 227 | -7.5% | 104 | 23 | 194 | -5.2% | 167 | 24 | 130 | +1.6% |
| Cold | 256 to 1023 tiles | 152 | 102 | 10 | 40 | +2.8% | 97 | 7 | 48 | +2.6% | 136 | 8 | 8 | +4.8% | 101 | 12 | 39 | +3.0% | 136 | 8 | 8 | +5.2% |
| Cold | 1024 or more tiles | 157 | 121 | 20 | 16 | +2.9% | 130 | 10 | 17 | +3.1% | 146 | 10 | 1 | +3.4% | 143 | 12 | 2 | +3.2% | 143 | 12 | 2 | +3.3% |
| Cold | Best library uses 256x256 | 262 | 208 | 27 | 27 | +2.9% | 220 | 17 | 25 | +3.2% | 246 | 15 | 1 | +3.6% | 227 | 20 | 15 | +3.2% | 241 | 18 | 3 | +3.5% |
| Cold | Best library uses another tile | 368 | 40 | 11 | 317 | -29.3% | 114 | 21 | 233 | -5.0% | 115 | 18 | 235 | -5.9% | 121 | 27 | 220 | -4.3% | 205 | 26 | 137 | +2.7% |

## Figure 1: progress over v9

![fig1](fig1_progress_cold.png)

- Cold: v9 248 ahead, 38 within 1%, 344 behind, median -3.9%; streamk_wip 446 ahead, 44 within 1%, 140 behind, median +3.4%. streamk_wip improved on v9 by more than 1% on 450 shapes and fell back on 39. 178 shapes went from behind to ahead, 1 the other way. hipBLASLt is the faster library on 385 shapes; the best library uses 256x256 on 262.

## Figure 2: the same-tile losses, by S and k-steps

![fig2](fig2_same_tile_S_k_cold.png)

- Cold: 262 shapes where the best library uses 256x256; streamk_wip is behind on 3, 2 of them data-parallel and 1 in the shaded band (k <= 48 and S = 0 or S > 128).
  - In the band (56 shapes): against the best library 42 ahead, 13 within 1%, 1 behind, median +1.8%; against v9 37 ahead, 17 within 1%, 2 behind, median +1.5%.
  - Outside it (206 shapes): against the best library 199 ahead, 5 within 1%, 2 behind, median +4.0%; against v9 91 ahead, 97 within 1%, 18 behind, median +0.7%.
  - Worst: 1024x2304x16384 (S = 36, k = 256, spread, -19.2% vs best, +152.4% vs v9); 768x129280x4096 (S = 235, k = 64, data-parallel, -3.4% vs best, -7.1% vs v9); 16384x7168x512 (S = 0, k = 8, data-parallel, -1.9% vs best, +7.0% vs v9).

## Figure 3: where the remaining losses are

![fig3](fig3_losses_cold.png)

- Cold: streamk_wip is behind the best library on 140 shapes.
  - Best library tile 256x256 (262 shapes): 241 ahead, 18 within 1%, 3 behind, median +3.5%. Losses by grid size (under 1 wave / 256-1023 / 1024+): 1 / 0 / 2.
  - Best library tile smaller, >= half the area (188 shapes): 120 ahead, 11 within 1%, 57 behind, median +4.3%. Losses by grid size (under 1 wave / 256-1023 / 1024+): 49 / 8 / 0.
  - Best library tile smaller, < half the area (180 shapes): 85 ahead, 15 within 1%, 80 behind, median +0.8%. Losses by grid size (under 1 wave / 256-1023 / 1024+): 80 / 0 / 0.

## Figure 4: shapes with no leftover tiles (S = 0)

![fig4](fig4_s0_by_waves_cold.png)

With S = 0 capped always runs data-parallel, so streamk_wip here is the persistent data-parallel kernel and stream-K plays no part.

- Cold: All S = 0 shapes: v9 +2.9% (18 behind), streamk_wip +3.0% (1 behind) of 181; 1 full wave: v9 +3.4% (0 behind), streamk_wip +3.3% (0 behind) of 27; 2-3 full waves: v9 +2.8% (2 behind), streamk_wip +3.1% (0 behind) of 33; 4-7 full waves: v9 +2.0% (9 behind), streamk_wip +2.7% (1 behind) of 46; 8 or more full waves: v9 +3.2% (7 behind), streamk_wip +2.6% (0 behind) of 75.

## Figure 5: shapes with leftover tiles (S > 0), by S

![fig5](fig5_s_buckets_cold.png)

Rows are S buckets; under one wave and one wave or more are separate panels. Data-parallel only to streamk_wip is stream-K's own effect.

## Figure 6: shapes with leftover tiles (S > 0), by S and k-steps

![fig6](fig6_s_k_heatmap_cold.png)

Each cell is the median over the shapes in it (n), with the number more than 1% behind the best library. Left: streamk_wip against the best library. Right: stream-K's gain, streamk_wip against the same kernel forced data-parallel; 0.0% means capped ran data-parallel on every shape in the cell.
