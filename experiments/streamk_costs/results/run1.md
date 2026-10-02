HIP_VISIBLE_DEVICES=2 M=N=4096 K=8192 launches/point=100

## check
  baseline C == A @ B: ok
  2 segments, switch=none: C == second half of K: ok
  2 segments, switch=c: C == second half of K: ok
  2 segments, switch=partial: C == second half of K: ok
  partial store, P layout 0: P[spid] == fp32 tile: ok
  partial store, P layout 1: P[spid] == fp32 tile: ok
  fixup s=2 store '.wt' read '.cv' layout 0 depth 1: owner C == sum of group tiles; flags re-armed: ok
  fixup s=4 store '.wt' read '.cv' layout 0 depth 1: owner C == sum of group tiles; flags re-armed: ok
  fixup s=16 store '.wt' read '.cv' layout 0 depth 1: owner C == sum of group tiles; flags re-armed: ok
  fixup s=16 store '' read '' layout 0 depth 1: owner C == sum of group tiles; flags re-armed: ok
  fixup s=16 store '.wt' read '.cv' layout 1 depth 1: owner C == sum of group tiles; flags re-armed: ok
  fixup s=8 store '' read '' layout 1 depth 1: owner C == sum of group tiles; flags re-armed: ok
  registers K=8192 base S=1: VGPR 488 AGPR 256 spill 0/0 scratch 0/0/0
  registers K=8192 store S=1: VGPR 512 AGPR 256 spill 16/78 scratch 0/16/16
  registers K=8192 fixup S=1: VGPR 512 AGPR 256 spill 26/38 scratch 0/22/10
  registers K=8192 switch S=2: VGPR 512 AGPR 256 spill 9/25 scratch 0/0/18

## kstep: baseline time vs K (one tile per program)

| K | k-steps | us |
|---|---|---|
| 1024 | 16 | 33.1 |
| 2048 | 32 | 56.4 |
| 4096 | 64 | 103.4 |
| 6144 | 96 | 141.7 |
| 8192 | 128 | 179.5 |
| 10240 | 160 | 216.9 |
| 12288 | 192 | 260.0 |

k-step = 1.269 us, fixed = 16.9 us, max residual 5.37 us  (VGPR 488 AGPR 256 spill 0/0 scratch 0/0/0)

## store: r partials (256 KiB each) per storer, then release flag; K=8192
(one partial per program is the stream-K case; r = 4 with 256 storers is 256 MiB and spills the MALL)

| storers | P layout | store cache | r: us | first partial: us (k-steps) | slope: us per partial | k-steps | max residual (us) |
|---|---|---|---|---|---|---|---|
| 256 | row-major | .wt | 0: 182.9, 1: 197.6, 2: 211.0, 4: 262.4 | 14.78 (11.65) | 19.99 | 15.76 | 7.49 |
| 240 | row-major | .wt | 0: 183.3, 1: 191.1, 2: 209.3, 4: 267.7 | 7.78 (6.13) | 21.76 | 17.15 | 8.98 |
| 16 | row-major | .wt | 0: 184.6, 1: 187.5, 2: 190.5, 4: 205.2 | 2.88 (2.27) | 5.21 | 4.11 | 2.73 |
| 256 | lane | .wt | 0: 183.1, 1: 192.6, 2: 208.9, 4: 241.6 | 9.52 (7.50) | 14.97 | 11.80 | 2.73 |
| 240 | lane | .wt | 0: 185.6, 1: 190.8, 2: 207.2, 4: 241.9 | 5.20 (4.10) | 14.64 | 11.54 | 4.85 |
| 16 | lane | .wt | 0: 184.1, 1: 185.0, 2: 185.6, 4: 190.9 | 0.92 (0.72) | 1.70 | 1.34 | 1.19 |
| 256 | lane | default | 0: 180.3, 1: 193.3, 2: 213.9, 4: 242.9 | 13.06 (10.29) | 15.95 | 12.57 | 2.30 |
| 240 | lane | default | 0: 179.7, 1: 196.3, 2: 211.7, 4: 245.6 | 16.64 (13.12) | 16.44 | 12.96 | 0.76 |
| 16 | lane | default | 0: 181.6, 1: 185.8, 2: 186.0, 4: 189.2 | 4.20 (3.31) | 1.70 | 1.34 | 1.46 |

(VGPR 512 AGPR 256 spill 16/78 scratch 0/16/16)

## read: owner reads n partials into its accumulator; K=8192
(read-only: partials host-filled, flags preset. fits use n >= 1: n = 0 skips the owner's per-quadrant setup)

| case | P layout | read cache | n: us | first partial: us (k-steps) | slope: us per partial | k-steps | max residual (us) |
|---|---|---|---|---|---|---|---|
| read-only, 16 owners | row-major | .cv | 0: 181.2, 1: 187.6, 2: 192.0, 4: 204.7, 8: 227.2, 12: 256.0, 15: 292.0 | 6.38 (5.03) | 7.15 | 5.63 | 8.21 |
| read-only, 16 owners | lane | .cv | 0: 183.5, 1: 186.2, 2: 188.6, 4: 188.5, 8: 195.9, 12: 216.7, 15: 237.0 | 2.66 (2.10) | 3.44 | 2.71 | 9.68 |
| read-only, 16 owners | lane | default | 0: 181.6, 1: 184.2, 2: 186.0, 4: 188.9, 8: 193.6, 12: 215.2, 15: 238.9 | 2.54 (2.00) | 3.62 | 2.85 | 11.19 |
| read-only, 16 owners, no flag polling | lane | .cv | 0: 183.2, 1: 183.1, 2: 187.6, 4: 188.7, 8: 195.9, 12: 214.9, 15: 235.5 | -0.14 (-0.11) | 3.45 | 2.72 | 8.49 |
| read-only, 256 owners | row-major | .cv | 0: 183.6, 1: 195.9, 2: 198.0, 4: 208.3, 8: 234.3, 12: 261.9, 15: 281.4 | 12.30 (9.70) | 6.28 | 4.95 | 3.64 |
| read-only, 256 owners | lane | .cv | 0: 183.9, 1: 191.2, 2: 194.8, 4: 195.3, 8: 203.6, 12: 217.7, 15: 221.2 | 7.30 (5.75) | 2.22 | 1.75 | 2.65 |
| fixup, groups of s (x = s-1 peers) | row-major | .wt / .cv | 0: 182.3, 1: 200.0, 3: 210.5, 7: 237.7, 15: 286.2 | 17.70 (13.95) | 6.22 | 4.90 | 1.32 |
| fixup, groups of s (x = s-1 peers) | lane | .wt / .cv | 0: 180.6, 1: 194.7, 3: 201.4, 7: 211.3, 15: 226.9 | 14.08 (11.10) | 2.25 | 1.78 | 1.60 |
| fixup, groups of s (x = s-1 peers) | lane | default / default | 0: 180.5, 1: 192.6, 3: 201.7, 7: 212.4, 15: 226.2 | 12.16 (9.58) | 2.29 | 1.81 | 3.05 |

(VGPR 512 AGPR 256 spill 26/38 scratch 0/22/10)

## switch: each tile runs as S segments of 8192/S; x = S-1 switches per program
(partial: fp32 .wt store + drain + release flag, lane layout unless noted)

| mode | switchers | S-1: us | first switch: us (k-steps) | slope: us per switch | k-steps | max residual (us) |
|---|---|---|---|---|---|---|
| none | 256 | 0: 182.4, 1: 182.3, 3: 185.7, 7: 191.9 | -0.12 (-0.09) | 1.44 | 1.13 | 0.81 |
| c | 256 | 0: 182.8, 1: 186.0, 3: 195.7, 7: 224.4 | 3.20 (2.52) | 6.06 | 4.78 | 3.02 |
| partial, row-major | 256 | 0: 181.9, 1: 198.0, 3: 237.7, 7: 327.7 | 16.02 (12.63) | 21.06 | 16.60 | 3.86 |
| partial | 256 | 0: 179.8, 1: 197.0, 3: 236.1, 7: 320.8 | 17.24 (13.59) | 20.29 | 15.99 | 2.43 |
| partial, relaxed flag | 256 | 0: 179.5, 1: 194.5, 3: 219.5, 7: 273.0 | 15.02 (11.84) | 13.26 | 10.45 | 1.09 |
| partial, default cache | 256 | 0: 181.4, 1: 199.3, 3: 238.9, 7: 328.8 | 17.96 (14.16) | 21.21 | 16.71 | 3.53 |
| c | 16 | 0: 181.2, 1: 185.3, 3: 191.9, 7: 206.3 | 4.08 (3.22) | 3.55 | 2.80 | 0.33 |
| partial, row-major | 16 | 0: 183.1, 1: 189.3, 3: 203.1, 7: 235.6 | 6.22 (4.90) | 7.56 | 5.96 | 1.56 |
| partial | 16 | 0: 183.5, 1: 185.1, 3: 197.6, 7: 211.5 | 1.60 (1.26) | 4.16 | 3.28 | 2.12 |
| partial, relaxed flag | 16 | 0: 182.3, 1: 185.3, 3: 193.9, 7: 209.2 | 3.06 (2.41) | 3.90 | 3.08 | 0.52 |
| partial, default cache | 16 | 0: 183.7, 1: 183.9, 3: 194.2, 7: 207.6 | 0.28 (0.22) | 3.61 | 2.85 | 2.08 |
(S=1: VGPR 512 AGPR 256 spill 16/78 scratch 0/16/16)
(S=2: VGPR 512 AGPR 256 spill 9/25 scratch 0/0/18)
(S=4: VGPR 512 AGPR 256 spill 9/18 scratch 0/0/18)
(S=8: VGPR 512 AGPR 256 spill 9/18 scratch 0/0/18)
