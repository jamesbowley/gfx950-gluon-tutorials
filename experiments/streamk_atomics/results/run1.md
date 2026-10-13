HIP_VISIBLE_DEVICES=0 launches/point=20 k-step=1.21 us

## check

- store: P[i] == src[i]: ok
- load: src[i] + P[i] (sink store) is right: ok
- c_store: C tile == src tile: ok
- owner: C tile == src tile + P tile: ok
- atom_lane, 1 writers per tile, x3: W[t] == sum of the writers' tiles: ok
- atom_coal, 1 writers per tile, x3: W[t] == sum of the writers' tiles: ok
- c_atom_acc, 1 writers per tile, x3: C tile == sum (bf16): ok
- c_atom_coal, 1 writers per tile, x3: C tile == sum (bf16): ok
- atom_lane, 2 writers per tile, x3: W[t] == sum of the writers' tiles: ok
- atom_coal, 2 writers per tile, x3: W[t] == sum of the writers' tiles: ok
- c_atom_acc, 2 writers per tile, x3: C tile == sum (bf16): ok
- c_atom_coal, 2 writers per tile, x3: C tile == sum (bf16): ok
- 2 writers per tile run on 2 different XCDs for 128/128 tiles: ok
- atom_lane, 4 writers per tile, x3: W[t] == sum of the writers' tiles: ok
- atom_coal, 4 writers per tile, x3: W[t] == sum of the writers' tiles: ok
- c_atom_acc, 4 writers per tile, x3: C tile == sum (bf16): ok
- c_atom_coal, 4 writers per tile, x3: C tile == sum (bf16): ok
- 4 writers per tile run on 4 different XCDs for 64/64 tiles: ok
- pair x3: C tile == contributor's bf16 store + owner's pk_add: ok
- pair_today x3: C tile == owner's tile + contributor's fp32 partial: ok

## asm

Per-thread instruction counts in the kernel (all four quadrants, including the 4 x 16 buffer_load_dwordx4 of the synthetic accumulator). 'returns' counts atomics with sc0 (the returning form).

| mode | VGPRs | spills | buffer_atomic_add_f32 | buffer_atomic_pk_add_bf16 | buffer_store_dwordx4 | buffer_store_dwordx2 | buffer_store_dword | buffer_load_dwordx4 | ds_write | ds_read | returns |
|---|---|---|---|---|---|---|---|---|---|---|---|
| store | 276 | 0 | 0 | 0 | 64 | 0 | 0 | 64 | 1 | 1 | 0 |
| load | 344 | 0 | 0 | 0 | 64 | 0 | 0 | 128 | 1 | 1 | 0 |
| atom_lane | 292 | 0 | 256 | 0 | 0 | 0 | 0 | 64 | 1 | 1 | 0 |
| atom_coal | 324 | 0 | 256 | 0 | 0 | 0 | 0 | 64 | 1 | 1 | 0 |
| c_store | 264 | 0 | 0 | 0 | 32 | 0 | 0 | 64 | 33 | 33 | 0 |
| c_atom_acc | 292 | 0 | 0 | 128 | 0 | 0 | 0 | 64 | 1 | 1 | 0 |
| c_atom_coal | 276 | 0 | 0 | 128 | 0 | 0 | 0 | 64 | 33 | 33 | 0 |
| pair | 340 | 0 | 0 | 128 | 32 | 0 | 0 | 64 | 66 | 66 | 0 |
| owner | 292 | 0 | 0 | 0 | 32 | 0 | 0 | 128 | 33 | 33 | 0 |
| pair_today | 348 | 0 | 0 | 0 | 96 | 0 | 0 | 128 | 34 | 34 | 0 |

Sample atomics:

    buffer_atomic_add_f32 v246, v15, s[4:7], 0 offen
    buffer_atomic_pk_add_bf16 v11, v12, s[4:7], 0 offen

## main: one ending per active program (warm caches)

| mode | writers per tile | active | KiB per program | median us | p90 us | median k-steps | span us | GB/s | start skew us |
|---|---|---|---|---|---|---|---|---|---|
| store | 1 | 1 | 256 | 1.88 | 1.88 | 1.55 | 1.88 | 139 | 0.00 |
| store | 1 | 2 | 256 | 1.84 | 1.88 | 1.52 | 1.97 | 266 | 0.13 |
| store | 1 | 4 | 256 | 1.88 | 1.88 | 1.55 | 2.02 | 519 | 0.15 |
| store | 1 | 16 | 256 | 1.92 | 1.96 | 1.59 | 2.17 | 1928 | 0.24 |
| store | 1 | 64 | 256 | 2.40 | 2.52 | 1.98 | 2.77 | 6068 | 0.26 |
| store | 1 | 256 | 256 | 6.64 | 7.88 | 5.49 | 8.20 | 8189 | 0.28 |
| load | 1 | 1 | 256 | 1.88 | 1.92 | 1.55 | 1.88 | 139 | 0.00 |
| load | 1 | 2 | 256 | 1.84 | 1.88 | 1.52 | 1.97 | 266 | 0.13 |
| load | 1 | 4 | 256 | 1.84 | 1.84 | 1.52 | 2.00 | 524 | 0.16 |
| load | 1 | 16 | 256 | 1.84 | 1.88 | 1.52 | 2.09 | 2007 | 0.23 |
| load | 1 | 64 | 256 | 2.04 | 2.08 | 1.69 | 2.34 | 7154 | 0.27 |
| load | 1 | 256 | 256 | 8.36 | 9.00 | 6.91 | 9.38 | 7154 | 0.26 |
| owner | 1 | 1 | 384 | 5.00 | 5.08 | 4.13 | 5.00 | 79 | 0.00 |
| owner | 1 | 2 | 384 | 5.00 | 5.08 | 4.13 | 5.09 | 155 | 0.11 |
| owner | 1 | 4 | 384 | 5.04 | 5.08 | 4.17 | 5.18 | 304 | 0.15 |
| owner | 1 | 16 | 384 | 5.04 | 5.12 | 4.17 | 5.28 | 1192 | 0.25 |
| owner | 1 | 64 | 384 | 6.16 | 6.36 | 5.09 | 6.59 | 3819 | 0.24 |
| owner | 1 | 256 | 384 | 13.24 | 14.20 | 10.94 | 14.70 | 6850 | 0.26 |
| c_store | 1 | 1 | 128 | 2.64 | 2.72 | 2.18 | 2.64 | 50 | 0.00 |
| c_store | 1 | 2 | 128 | 2.60 | 2.64 | 2.15 | 2.73 | 96 | 0.13 |
| c_store | 1 | 4 | 128 | 2.60 | 2.64 | 2.15 | 2.78 | 189 | 0.15 |
| c_store | 1 | 16 | 128 | 2.60 | 2.64 | 2.15 | 2.84 | 737 | 0.23 |
| c_store | 1 | 64 | 128 | 2.68 | 2.72 | 2.21 | 2.91 | 2888 | 0.22 |
| c_store | 1 | 256 | 128 | 3.56 | 3.92 | 2.94 | 4.17 | 8056 | 0.27 |
| atom_lane | 1 | 1 | 256 | 13.14 | 13.28 | 10.86 | 13.14 | 20 | 0.00 |
| atom_lane | 1 | 2 | 256 | 13.02 | 13.48 | 10.76 | 13.36 | 39 | 0.11 |
| atom_lane | 1 | 4 | 256 | 13.22 | 13.64 | 10.93 | 13.68 | 77 | 0.15 |
| atom_lane | 1 | 16 | 256 | 16.76 | 17.60 | 13.85 | 17.81 | 236 | 0.23 |
| atom_lane | 1 | 64 | 256 | 49.92 | 53.48 | 41.26 | 54.15 | 310 | 0.25 |
| atom_lane | 1 | 256 | 256 | 191.76 | 208.40 | 158.48 | 211.27 | 318 | 0.28 |
| atom_lane | 2 | 2 | 256 | 13.04 | 13.40 | 10.78 | 13.30 | 39 | 0.13 |
| atom_lane | 2 | 4 | 256 | 13.60 | 14.08 | 11.24 | 14.07 | 74 | 0.14 |
| atom_lane | 2 | 16 | 256 | 17.44 | 18.28 | 14.41 | 18.41 | 228 | 0.25 |
| atom_lane | 2 | 64 | 256 | 50.16 | 54.36 | 41.45 | 55.08 | 305 | 0.23 |
| atom_lane | 2 | 256 | 256 | 190.12 | 206.80 | 157.12 | 209.99 | 320 | 0.28 |
| atom_lane | 4 | 4 | 256 | 13.60 | 14.12 | 11.24 | 14.14 | 74 | 0.14 |
| atom_lane | 4 | 16 | 256 | 17.12 | 18.00 | 14.15 | 18.16 | 231 | 0.23 |
| atom_lane | 4 | 64 | 256 | 50.00 | 53.56 | 41.32 | 54.01 | 311 | 0.23 |
| atom_lane | 4 | 256 | 256 | 186.32 | 205.32 | 153.98 | 209.05 | 321 | 0.26 |
| atom_coal | 1 | 1 | 256 | 7.24 | 7.28 | 5.98 | 7.24 | 36 | 0.00 |
| atom_coal | 1 | 2 | 256 | 7.20 | 7.24 | 5.95 | 7.33 | 72 | 0.13 |
| atom_coal | 1 | 4 | 256 | 7.16 | 7.24 | 5.92 | 7.35 | 143 | 0.15 |
| atom_coal | 1 | 16 | 256 | 7.20 | 7.24 | 5.95 | 7.45 | 563 | 0.24 |
| atom_coal | 1 | 64 | 256 | 12.64 | 12.76 | 10.45 | 12.96 | 1295 | 0.20 |
| atom_coal | 1 | 256 | 256 | 47.76 | 49.40 | 39.47 | 49.77 | 1348 | 0.28 |
| atom_coal | 2 | 2 | 256 | 7.20 | 7.24 | 5.95 | 7.33 | 72 | 0.13 |
| atom_coal | 2 | 4 | 256 | 7.20 | 7.24 | 5.95 | 7.35 | 143 | 0.14 |
| atom_coal | 2 | 16 | 256 | 7.20 | 7.24 | 5.95 | 7.46 | 563 | 0.22 |
| atom_coal | 2 | 64 | 256 | 12.64 | 12.72 | 10.45 | 13.02 | 1289 | 0.20 |
| atom_coal | 2 | 256 | 256 | 47.70 | 49.44 | 39.42 | 49.75 | 1349 | 0.25 |
| atom_coal | 4 | 4 | 256 | 7.20 | 7.24 | 5.95 | 7.39 | 142 | 0.15 |
| atom_coal | 4 | 16 | 256 | 7.20 | 7.24 | 5.95 | 7.45 | 563 | 0.24 |
| atom_coal | 4 | 64 | 256 | 12.60 | 12.72 | 10.41 | 12.92 | 1299 | 0.23 |
| atom_coal | 4 | 256 | 256 | 46.92 | 49.40 | 38.78 | 49.76 | 1349 | 0.24 |
| c_atom_acc | 1 | 1 | 128 | 15.16 | 15.28 | 12.53 | 15.16 | 9 | 0.00 |
| c_atom_acc | 1 | 2 | 128 | 14.50 | 15.20 | 11.98 | 15.16 | 17 | 0.13 |
| c_atom_acc | 1 | 4 | 128 | 14.52 | 15.20 | 12.00 | 15.26 | 34 | 0.14 |
| c_atom_acc | 1 | 16 | 128 | 14.66 | 15.32 | 12.12 | 15.49 | 135 | 0.22 |
| c_atom_acc | 1 | 64 | 128 | 24.74 | 26.16 | 20.45 | 26.71 | 314 | 0.22 |
| c_atom_acc | 1 | 256 | 128 | 96.96 | 99.08 | 80.13 | 99.71 | 337 | 0.28 |
| c_atom_acc | 2 | 2 | 128 | 14.90 | 15.16 | 12.31 | 15.16 | 17 | 0.12 |
| c_atom_acc | 2 | 4 | 128 | 14.34 | 15.16 | 11.85 | 15.23 | 34 | 0.14 |
| c_atom_acc | 2 | 16 | 128 | 14.68 | 15.88 | 12.13 | 15.99 | 131 | 0.23 |
| c_atom_acc | 2 | 64 | 128 | 24.60 | 26.00 | 20.33 | 26.61 | 315 | 0.24 |
| c_atom_acc | 2 | 256 | 128 | 95.32 | 99.04 | 78.78 | 100.08 | 335 | 0.26 |
| c_atom_acc | 4 | 4 | 128 | 14.94 | 15.24 | 12.35 | 15.32 | 34 | 0.14 |
| c_atom_acc | 4 | 16 | 128 | 15.86 | 17.04 | 13.11 | 17.32 | 121 | 0.24 |
| c_atom_acc | 4 | 64 | 128 | 24.60 | 25.92 | 20.33 | 26.54 | 316 | 0.25 |
| c_atom_acc | 4 | 256 | 128 | 95.42 | 98.60 | 78.86 | 99.52 | 337 | 0.26 |
| c_atom_coal | 1 | 1 | 128 | 4.80 | 4.92 | 3.97 | 4.80 | 27 | 0.00 |
| c_atom_coal | 1 | 2 | 128 | 4.76 | 4.84 | 3.93 | 4.85 | 54 | 0.13 |
| c_atom_coal | 1 | 4 | 128 | 4.72 | 4.80 | 3.90 | 4.95 | 106 | 0.15 |
| c_atom_coal | 1 | 16 | 128 | 4.76 | 4.84 | 3.93 | 5.00 | 420 | 0.22 |
| c_atom_coal | 1 | 64 | 128 | 6.72 | 6.88 | 5.55 | 7.05 | 1190 | 0.22 |
| c_atom_coal | 1 | 256 | 128 | 23.88 | 25.16 | 19.74 | 25.42 | 1320 | 0.26 |
| c_atom_coal | 2 | 2 | 128 | 4.68 | 4.80 | 3.87 | 4.80 | 55 | 0.11 |
| c_atom_coal | 2 | 4 | 128 | 4.72 | 4.76 | 3.90 | 4.87 | 108 | 0.15 |
| c_atom_coal | 2 | 16 | 128 | 4.76 | 4.80 | 3.93 | 4.98 | 421 | 0.24 |
| c_atom_coal | 2 | 64 | 128 | 6.68 | 6.80 | 5.52 | 6.95 | 1207 | 0.22 |
| c_atom_coal | 2 | 256 | 128 | 24.12 | 25.16 | 19.93 | 25.38 | 1322 | 0.27 |
| c_atom_coal | 4 | 4 | 128 | 4.68 | 4.76 | 3.87 | 4.87 | 108 | 0.15 |
| c_atom_coal | 4 | 16 | 128 | 4.80 | 4.88 | 3.97 | 5.02 | 418 | 0.22 |
| c_atom_coal | 4 | 64 | 128 | 6.64 | 6.76 | 5.49 | 6.94 | 1209 | 0.23 |
| c_atom_coal | 4 | 256 | 128 | 23.76 | 25.08 | 19.64 | 25.31 | 1326 | 0.26 |

## pair: S tiles, one contributor and one owner each (warm caches)

today (pair_today): the contributor stores its fp32 partial (lane-contiguous), drains and raises a release flag; the owner polls, acquires, reads the partial, adds and stores bf16 C.
atomic (pair): the contributor stores its partial as bf16 into C, drains and raises the flag; the owner polls, acquires and pk_adds its own tile into C.
Both start with the accumulator ready on both programs, so 'whole' is the fixup's critical path when neither program has MAC work left. us (k-steps).

| S | today: publish | today: wait | today: owner ending | today: whole | atomic: publish | atomic: wait | atomic: owner ending | atomic: whole |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.24 (2.68) | 3.70 (3.06) | 6.32 (5.22) | 10.08 (8.33) | 3.68 (3.04) | 4.18 (3.45) | 4.70 (3.88) | 8.93 (7.38) |
| 2 | 3.28 (2.71) | 3.78 (3.12) | 6.26 (5.17) | 10.08 (8.33) | 3.64 (3.01) | 4.16 (3.44) | 4.76 (3.93) | 8.92 (7.37) |
| 4 | 3.20 (2.64) | 3.76 (3.11) | 6.12 (5.06) | 9.93 (8.20) | 3.60 (2.98) | 4.18 (3.45) | 4.76 (3.93) | 9.00 (7.43) |
| 8 | 3.76 (3.11) | 4.32 (3.57) | 6.14 (5.07) | 10.52 (8.69) | 3.76 (3.11) | 4.36 (3.60) | 4.80 (3.97) | 9.17 (7.58) |
| 16 | 4.96 (4.10) | 5.68 (4.69) | 6.16 (5.09) | 11.84 (9.79) | 4.08 (3.37) | 4.80 (3.97) | 4.80 (3.97) | 9.64 (7.97) |
| 32 | 7.96 (6.58) | 8.76 (7.24) | 6.32 (5.22) | 15.08 (12.46) | 4.76 (3.93) | 5.64 (4.66) | 4.80 (3.97) | 10.48 (8.66) |
| 64 | 6.52 (5.39) | 7.64 (6.31) | 7.24 (5.98) | 14.97 (12.37) | 6.00 (4.96) | 7.16 (5.92) | 6.36 (5.26) | 13.64 (11.27) |
| 128 | 11.64 (9.62) | 12.80 (10.58) | 9.28 (7.67) | 22.24 (18.38) | 8.20 (6.78) | 9.60 (7.93) | 11.56 (9.55) | 21.44 (17.72) |
