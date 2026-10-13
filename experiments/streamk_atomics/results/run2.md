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

## main: one ending per active program (warm caches)

| mode | writers per tile | active | KiB per program | median us | p90 us | median k-steps | span us | GB/s | start skew us |
|---|---|---|---|---|---|---|---|---|---|
| store | 1 | 1 | 256 | 1.88 | 1.92 | 1.55 | 1.88 | 139 | 0.00 |
| store | 1 | 2 | 256 | 1.88 | 1.88 | 1.55 | 1.91 | 274 | 0.05 |
| store | 1 | 4 | 256 | 1.88 | 1.88 | 1.55 | 2.00 | 523 | 0.14 |
| store | 1 | 16 | 256 | 1.92 | 2.00 | 1.59 | 2.17 | 1933 | 0.22 |
| store | 1 | 64 | 256 | 2.44 | 2.56 | 2.02 | 2.74 | 6134 | 0.23 |
| store | 1 | 256 | 256 | 6.68 | 7.88 | 5.52 | 8.17 | 8214 | 0.24 |
| load | 1 | 1 | 256 | 1.88 | 1.92 | 1.55 | 1.88 | 139 | 0.00 |
| load | 1 | 2 | 256 | 1.84 | 1.88 | 1.52 | 1.88 | 279 | 0.04 |
| load | 1 | 4 | 256 | 1.84 | 1.84 | 1.52 | 1.98 | 530 | 0.14 |
| load | 1 | 16 | 256 | 1.84 | 1.88 | 1.52 | 2.09 | 2002 | 0.23 |
| load | 1 | 64 | 256 | 2.04 | 2.12 | 1.69 | 2.33 | 7201 | 0.23 |
| load | 1 | 256 | 256 | 8.40 | 8.96 | 6.94 | 9.28 | 7232 | 0.24 |
| owner | 1 | 1 | 384 | 5.00 | 5.08 | 4.13 | 5.00 | 79 | 0.00 |
| owner | 1 | 2 | 384 | 5.00 | 5.08 | 4.13 | 5.04 | 156 | 0.03 |
| owner | 1 | 4 | 384 | 5.00 | 5.08 | 4.13 | 5.11 | 308 | 0.10 |
| owner | 1 | 16 | 384 | 5.04 | 5.08 | 4.17 | 5.27 | 1195 | 0.19 |
| owner | 1 | 64 | 384 | 6.16 | 6.36 | 5.09 | 6.62 | 3804 | 0.22 |
| owner | 1 | 256 | 384 | 13.20 | 14.24 | 10.91 | 14.79 | 6808 | 0.25 |
| c_store | 1 | 1 | 128 | 2.68 | 2.72 | 2.21 | 2.68 | 49 | 0.00 |
| c_store | 1 | 2 | 128 | 2.60 | 2.64 | 2.15 | 2.64 | 99 | 0.03 |
| c_store | 1 | 4 | 128 | 2.60 | 2.64 | 2.15 | 2.78 | 189 | 0.14 |
| c_store | 1 | 16 | 128 | 2.60 | 2.64 | 2.15 | 2.83 | 741 | 0.21 |
| c_store | 1 | 64 | 128 | 2.68 | 2.72 | 2.21 | 2.93 | 2863 | 0.24 |
| c_store | 1 | 256 | 128 | 3.52 | 3.92 | 2.91 | 4.16 | 8076 | 0.25 |
| atom_lane | 1 | 1 | 256 | 13.64 | 13.84 | 11.27 | 13.64 | 19 | 0.00 |
| atom_lane | 1 | 2 | 256 | 13.60 | 14.00 | 11.24 | 13.88 | 38 | 0.03 |
| atom_lane | 1 | 4 | 256 | 13.88 | 14.36 | 11.47 | 14.26 | 74 | 0.14 |
| atom_lane | 1 | 16 | 256 | 16.90 | 18.16 | 13.97 | 18.22 | 230 | 0.22 |
| atom_lane | 1 | 64 | 256 | 50.48 | 54.48 | 41.72 | 54.97 | 305 | 0.23 |
| atom_lane | 1 | 256 | 256 | 190.28 | 206.28 | 157.26 | 208.10 | 322 | 0.25 |
| atom_lane | 2 | 2 | 256 | 13.44 | 14.00 | 11.11 | 13.87 | 38 | 0.03 |
| atom_lane | 2 | 4 | 256 | 14.04 | 14.52 | 11.60 | 14.50 | 72 | 0.12 |
| atom_lane | 2 | 16 | 256 | 17.44 | 18.40 | 14.41 | 18.39 | 228 | 0.22 |
| atom_lane | 2 | 64 | 256 | 49.88 | 54.00 | 41.22 | 54.41 | 308 | 0.23 |
| atom_lane | 2 | 256 | 256 | 191.16 | 206.52 | 157.98 | 208.88 | 321 | 0.26 |
| atom_lane | 4 | 4 | 256 | 14.04 | 14.48 | 11.60 | 14.41 | 73 | 0.12 |
| atom_lane | 4 | 16 | 256 | 17.24 | 18.28 | 14.25 | 18.27 | 230 | 0.21 |
| atom_lane | 4 | 64 | 256 | 50.20 | 53.84 | 41.49 | 54.29 | 309 | 0.23 |
| atom_lane | 4 | 256 | 256 | 183.64 | 202.60 | 151.77 | 205.31 | 327 | 0.25 |
| atom_coal | 1 | 1 | 256 | 7.20 | 7.20 | 5.95 | 7.20 | 36 | 0.00 |
| atom_coal | 1 | 2 | 256 | 7.12 | 7.16 | 5.88 | 7.16 | 73 | 0.03 |
| atom_coal | 1 | 4 | 256 | 7.12 | 7.16 | 5.88 | 7.22 | 145 | 0.10 |
| atom_coal | 1 | 16 | 256 | 7.20 | 7.28 | 5.95 | 7.46 | 562 | 0.21 |
| atom_coal | 1 | 64 | 256 | 12.64 | 12.76 | 10.45 | 12.96 | 1295 | 0.23 |
| atom_coal | 1 | 256 | 256 | 47.80 | 49.40 | 39.50 | 49.73 | 1349 | 0.23 |
| atom_coal | 2 | 2 | 256 | 7.12 | 7.16 | 5.88 | 7.15 | 73 | 0.03 |
| atom_coal | 2 | 4 | 256 | 7.12 | 7.16 | 5.88 | 7.22 | 145 | 0.14 |
| atom_coal | 2 | 16 | 256 | 7.20 | 7.28 | 5.95 | 7.48 | 561 | 0.21 |
| atom_coal | 2 | 64 | 256 | 12.64 | 12.76 | 10.45 | 13.03 | 1288 | 0.23 |
| atom_coal | 2 | 256 | 256 | 47.80 | 49.44 | 39.50 | 49.75 | 1349 | 0.24 |
| atom_coal | 4 | 4 | 256 | 7.12 | 7.16 | 5.88 | 7.26 | 144 | 0.14 |
| atom_coal | 4 | 16 | 256 | 7.16 | 7.20 | 5.92 | 7.42 | 565 | 0.21 |
| atom_coal | 4 | 64 | 256 | 12.60 | 12.72 | 10.41 | 12.89 | 1301 | 0.23 |
| atom_coal | 4 | 256 | 256 | 46.44 | 49.40 | 38.38 | 49.75 | 1349 | 0.25 |
| c_atom_acc | 1 | 1 | 128 | 15.20 | 15.28 | 12.56 | 15.20 | 9 | 0.00 |
| c_atom_acc | 1 | 2 | 128 | 14.54 | 15.24 | 12.02 | 15.19 | 17 | 0.03 |
| c_atom_acc | 1 | 4 | 128 | 14.56 | 15.24 | 12.03 | 15.24 | 34 | 0.14 |
| c_atom_acc | 1 | 16 | 128 | 14.68 | 15.40 | 12.13 | 15.68 | 134 | 0.21 |
| c_atom_acc | 1 | 64 | 128 | 24.72 | 26.16 | 20.43 | 26.79 | 313 | 0.23 |
| c_atom_acc | 1 | 256 | 128 | 96.88 | 98.96 | 80.07 | 99.44 | 337 | 0.26 |
| c_atom_acc | 2 | 2 | 128 | 14.88 | 15.20 | 12.30 | 15.16 | 17 | 0.03 |
| c_atom_acc | 2 | 4 | 128 | 14.32 | 15.16 | 11.83 | 15.23 | 34 | 0.14 |
| c_atom_acc | 2 | 16 | 128 | 14.68 | 15.84 | 12.13 | 16.01 | 131 | 0.22 |
| c_atom_acc | 2 | 64 | 128 | 24.68 | 26.12 | 20.40 | 26.71 | 314 | 0.22 |
| c_atom_acc | 2 | 256 | 128 | 95.40 | 99.52 | 78.84 | 100.42 | 334 | 0.24 |
| c_atom_acc | 4 | 4 | 128 | 14.98 | 15.28 | 12.38 | 15.30 | 34 | 0.14 |
| c_atom_acc | 4 | 16 | 128 | 15.88 | 17.08 | 13.12 | 17.27 | 121 | 0.23 |
| c_atom_acc | 4 | 64 | 128 | 24.80 | 26.00 | 20.50 | 26.44 | 317 | 0.23 |
| c_atom_acc | 4 | 256 | 128 | 95.20 | 98.76 | 78.68 | 99.59 | 337 | 0.25 |
| c_atom_coal | 1 | 1 | 128 | 4.84 | 4.92 | 4.00 | 4.84 | 27 | 0.00 |
| c_atom_coal | 1 | 2 | 128 | 4.76 | 4.88 | 3.93 | 4.84 | 54 | 0.03 |
| c_atom_coal | 1 | 4 | 128 | 4.72 | 4.80 | 3.90 | 4.89 | 107 | 0.14 |
| c_atom_coal | 1 | 16 | 128 | 4.80 | 4.88 | 3.97 | 5.06 | 414 | 0.22 |
| c_atom_coal | 1 | 64 | 128 | 6.76 | 6.92 | 5.59 | 7.08 | 1185 | 0.21 |
| c_atom_coal | 1 | 256 | 128 | 24.04 | 25.16 | 19.87 | 25.40 | 1321 | 0.26 |
| c_atom_coal | 2 | 2 | 128 | 4.70 | 4.80 | 3.88 | 4.75 | 55 | 0.03 |
| c_atom_coal | 2 | 4 | 128 | 4.68 | 4.76 | 3.87 | 4.83 | 109 | 0.13 |
| c_atom_coal | 2 | 16 | 128 | 4.76 | 4.84 | 3.93 | 4.99 | 420 | 0.21 |
| c_atom_coal | 2 | 64 | 128 | 6.60 | 6.76 | 5.45 | 6.95 | 1208 | 0.23 |
| c_atom_coal | 2 | 256 | 128 | 24.08 | 25.12 | 19.90 | 25.37 | 1323 | 0.27 |
| c_atom_coal | 4 | 4 | 128 | 4.68 | 4.76 | 3.87 | 4.83 | 109 | 0.11 |
| c_atom_coal | 4 | 16 | 128 | 4.80 | 4.88 | 3.97 | 5.03 | 417 | 0.22 |
| c_atom_coal | 4 | 64 | 128 | 6.68 | 6.84 | 5.52 | 6.95 | 1207 | 0.23 |
| c_atom_coal | 4 | 256 | 128 | 23.96 | 25.08 | 19.80 | 25.32 | 1325 | 0.25 |

## pair: S tiles, one contributor and one owner each (warm caches)

today (pair_today): the contributor stores its fp32 partial (lane-contiguous), drains and raises a release flag; the owner polls, acquires, reads the partial, adds and stores bf16 C.
atomic (pair): the contributor stores its partial as bf16 into C, drains and raises the flag; the owner polls, acquires and pk_adds its own tile into C.
Both start with the accumulator ready on both programs, so 'whole' is the fixup's critical path when neither program has MAC work left. us (k-steps).

| S | today: publish | today: wait | today: owner ending | today: whole | atomic: publish | atomic: wait | atomic: owner ending | atomic: whole |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.04 (2.51) | 3.52 (2.91) | 6.24 (5.16) | 9.75 (8.06) | 3.48 (2.88) | 3.96 (3.27) | 4.76 (3.93) | 8.70 (7.19) |
| 2 | 3.12 (2.58) | 3.56 (2.94) | 6.24 (5.16) | 9.86 (8.15) | 3.48 (2.88) | 3.92 (3.24) | 4.72 (3.90) | 8.71 (7.19) |
| 4 | 3.20 (2.64) | 3.70 (3.06) | 6.10 (5.04) | 9.87 (8.16) | 3.60 (2.98) | 4.16 (3.44) | 4.76 (3.93) | 8.96 (7.40) |
| 8 | 3.76 (3.11) | 4.32 (3.57) | 6.10 (5.04) | 10.52 (8.69) | 3.76 (3.11) | 4.40 (3.64) | 4.76 (3.93) | 9.21 (7.61) |
| 16 | 4.92 (4.07) | 5.72 (4.73) | 6.12 (5.06) | 11.88 (9.82) | 4.08 (3.37) | 4.88 (4.03) | 4.80 (3.97) | 9.68 (8.00) |
| 32 | 8.00 (6.61) | 8.68 (7.17) | 6.36 (5.26) | 15.07 (12.45) | 4.76 (3.93) | 5.60 (4.63) | 4.80 (3.97) | 10.46 (8.64) |
| 64 | 6.52 (5.39) | 7.64 (6.31) | 7.28 (6.02) | 15.00 (12.39) | 6.02 (4.98) | 7.16 (5.92) | 6.36 (5.26) | 13.63 (11.26) |
| 128 | 11.52 (9.52) | 12.64 (10.45) | 9.32 (7.70) | 22.14 (18.30) | 8.12 (6.71) | 9.60 (7.93) | 11.68 (9.65) | 21.59 (17.84) |
