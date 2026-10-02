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
| 1024 | 16 | 32.9 |
| 2048 | 32 | 56.6 |
| 4096 | 64 | 102.2 |
| 6144 | 96 | 139.7 |
| 8192 | 128 | 182.5 |
| 10240 | 160 | 218.1 |
| 12288 | 192 | 254.4 |

k-step = 1.256 us, fixed = 17.4 us, max residual 4.62 us  (VGPR 488 AGPR 256 spill 0/0 scratch 0/0/0)

## store: r partials (256 KiB each) per storer, then release flag; K=8192
(one partial per program is the stream-K case; r = 4 with 256 storers is 256 MiB and spills the MALL)

| storers | P layout | store cache | r: us | first partial: us (k-steps) | slope: us per partial | k-steps | max residual (us) |
|---|---|---|---|---|---|---|---|
| 256 | row-major | .wt | 0: 183.9, 1: 193.1, 2: 209.7, 4: 248.8 | 9.16 (7.29) | 16.65 | 13.25 | 4.16 |
| 240 | row-major | .wt | 0: 181.0, 1: 190.9, 2: 208.7, 4: 247.0 | 9.86 (7.85) | 16.90 | 13.45 | 3.71 |
| 16 | row-major | .wt | 0: 183.5, 1: 186.6, 2: 193.4, 4: 203.2 | 3.14 (2.50) | 5.08 | 4.04 | 1.24 |
| 256 | lane | .wt | 0: 180.6, 1: 194.0, 2: 209.0, 4: 234.5 | 13.42 (10.68) | 13.54 | 10.78 | 1.10 |
| 240 | lane | .wt | 0: 181.7, 1: 191.6, 2: 206.6, 4: 231.5 | 9.84 (7.83) | 12.66 | 10.08 | 1.78 |
| 16 | lane | .wt | 0: 183.0, 1: 185.1, 2: 192.5, 4: 191.0 | 2.14 (1.70) | 2.16 | 1.72 | 4.09 |
| 256 | lane | default | 0: 181.6, 1: 196.1, 2: 212.7, 4: 234.5 | 14.58 (11.61) | 13.26 | 10.55 | 3.14 |
| 240 | lane | default | 0: 179.3, 1: 195.0, 2: 209.2, 4: 235.4 | 15.68 (12.48) | 13.93 | 11.09 | 1.02 |
| 16 | lane | default | 0: 179.4, 1: 187.1, 2: 187.9, 4: 189.4 | 7.70 (6.13) | 2.15 | 1.71 | 2.79 |

(VGPR 512 AGPR 256 spill 16/78 scratch 0/16/16)

## read: owner reads n partials into its accumulator; K=8192
(read-only: partials host-filled, flags preset. fits use n >= 1: n = 0 skips the owner's per-quadrant setup)

| case | P layout | read cache | n: us | first partial: us (k-steps) | slope: us per partial | k-steps | max residual (us) |
|---|---|---|---|---|---|---|---|
| read-only, 16 owners | row-major | .cv | 0: 185.5, 1: 187.8, 2: 192.8, 4: 200.9, 8: 225.4, 12: 258.9, 15: 290.3 | 2.24 (1.78) | 7.18 | 5.72 | 7.80 |
| read-only, 16 owners | lane | .cv | 0: 182.7, 1: 181.5, 2: 188.6, 4: 188.2, 8: 194.1, 12: 215.0, 15: 236.3 | -1.20 (-0.96) | 3.52 | 2.80 | 10.02 |
| read-only, 16 owners | lane | default | 0: 181.8, 1: 184.8, 2: 184.9, 4: 189.5, 8: 196.4, 12: 218.1, 15: 236.2 | 2.96 (2.36) | 3.60 | 2.86 | 8.81 |
| read-only, 16 owners, no flag polling | lane | .cv | 0: 181.6, 1: 181.8, 2: 187.7, 4: 191.5, 8: 193.6, 12: 217.2, 15: 234.5 | 0.24 (0.19) | 3.45 | 2.74 | 10.86 |
| read-only, 256 owners | row-major | .cv | 0: 182.4, 1: 199.2, 2: 200.4, 4: 211.1, 8: 240.2, 12: 261.2, 15: 282.5 | 16.82 (13.39) | 6.10 | 4.86 | 3.35 |
| read-only, 256 owners | lane | .cv | 0: 178.5, 1: 190.5, 2: 195.6, 4: 199.2, 8: 209.8, 12: 216.2, 15: 223.2 | 12.06 (9.60) | 2.23 | 1.78 | 1.82 |
| fixup, groups of s (x = s-1 peers) | row-major | .wt / .cv | 0: 184.0, 1: 198.5, 3: 212.2, 7: 235.1, 15: 286.1 | 14.42 (11.48) | 6.22 | 4.95 | 1.02 |
| fixup, groups of s (x = s-1 peers) | lane | .wt / .cv | 0: 181.9, 1: 193.5, 3: 196.7, 7: 208.8, 15: 225.6 | 11.52 (9.17) | 2.34 | 1.87 | 1.44 |
| fixup, groups of s (x = s-1 peers) | lane | default / default | 0: 185.3, 1: 195.0, 3: 204.1, 7: 207.8, 15: 227.7 | 9.74 (7.75) | 2.19 | 1.75 | 3.15 |

(VGPR 512 AGPR 256 spill 26/38 scratch 0/22/10)

## switch: each tile runs as S segments of 8192/S; x = S-1 switches per program
(partial: fp32 .wt store + drain + release flag, lane layout unless noted)

| mode | switchers | S-1: us | first switch: us (k-steps) | slope: us per switch | k-steps | max residual (us) |
|---|---|---|---|---|---|---|
| none | 256 | 0: 182.6, 1: 183.8, 3: 186.7, 7: 193.2 | 1.24 (0.99) | 1.52 | 1.21 | 0.23 |
| c | 256 | 0: 184.6, 1: 189.3, 3: 196.2, 7: 224.3 | 4.76 (3.79) | 5.68 | 4.52 | 3.81 |
| partial, row-major | 256 | 0: 182.7, 1: 200.2, 3: 237.1, 7: 327.4 | 17.46 (13.90) | 20.79 | 16.55 | 4.96 |
| partial | 256 | 0: 180.8, 1: 199.7, 3: 234.0, 7: 317.3 | 18.90 (15.04) | 19.49 | 15.51 | 3.79 |
| partial, relaxed flag | 256 | 0: 180.2, 1: 194.6, 3: 222.9, 7: 267.8 | 14.44 (11.49) | 12.45 | 9.91 | 3.38 |
| partial, default cache | 256 | 0: 183.2, 1: 195.8, 3: 241.9, 7: 327.8 | 12.60 (10.03) | 21.13 | 16.82 | 4.43 |
| c | 16 | 0: 185.1, 1: 184.8, 3: 191.4, 7: 207.8 | -0.28 (-0.22) | 3.43 | 2.73 | 2.24 |
| partial, row-major | 16 | 0: 179.3, 1: 188.9, 3: 200.8, 7: 233.1 | 9.56 (7.61) | 7.55 | 6.01 | 1.60 |
| partial | 16 | 0: 181.4, 1: 183.8, 3: 191.6, 7: 211.0 | 2.42 (1.93) | 4.33 | 3.44 | 1.45 |
| partial, relaxed flag | 16 | 0: 182.9, 1: 186.2, 3: 192.3, 7: 205.9 | 3.30 (2.63) | 3.28 | 2.61 | 0.32 |
| partial, default cache | 16 | 0: 182.8, 1: 186.0, 3: 190.7, 7: 212.0 | 3.24 (2.58) | 4.19 | 3.33 | 3.20 |
(S=1: VGPR 512 AGPR 256 spill 16/78 scratch 0/16/16)
(S=2: VGPR 512 AGPR 256 spill 9/25 scratch 0/0/18)
(S=4: VGPR 512 AGPR 256 spill 9/18 scratch 0/0/18)
(S=8: VGPR 512 AGPR 256 spill 9/18 scratch 0/0/18)
