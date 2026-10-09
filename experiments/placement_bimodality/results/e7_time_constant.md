# E7: idle gaps, writes from another kernel, and the offset sweep with gaps

v9, 4096x4096x1024, bf16, GPU 2. One 1 GB buffer: A/B at 900 MB, C at 32 MB (fast half, E2)
or 160 MB (slow half). After each GEMM the graph either idles (`torch.cuda._sleep`, calibrated
and subtracted) or runs a `fill_` of some size somewhere in the buffer (its own graph time
subtracted). 40 replays of 400 GEMMs; MHz and W are hwmon medians while it replays. The power
readings here are flatter than in E1 (1283-1292 W with no gap); 40 replays may be too short
for the power average to settle, so they are not used. Raw: `e7_raw.txt`,
`e7_sweep_gap{0,20,100}.txt`.

## Idle gap after each GEMM (`place.py e7`)

| idle gap (us) | fast half us | MHz | slow half us | MHz | slow - fast (us) |
|---|---|---|---|---|---|
| 0 | 28.32 | 2069 | 29.48 | 2012 | 1.16 |
| 2 | 28.52 | 2111 | 29.79 | 2055 | 1.27 |
| 5 | 28.40 | 2147 | 29.69 | 2094 | 1.29 |
| 10 | 29.07 | 2147 | 29.93 | 2102 | 0.86 |
| 20 | 28.05 | 2250 | 29.71 | 2193 | 1.66 |
| 50 | 27.86 | 2305 | 29.10 | 2287 | 1.24 |
| 100 | 27.24 | 2354 | 27.46 | 2354 | 0.22 |
| 200 | 26.40 | 2387 | 26.39 | 2387 | -0.01 |

## The whole offset sweep with gaps (`sweep --what c --step 16 --gap G`)

| idle gap (us) | fast half (C at 0 - 96 MB), median | slow half (C at 128 - 224 MB), median | MHz fast / slow |
|---|---|---|---|
| 0 | 28.31 | 29.56 | not sampled |
| 20 | 28.03 | 29.67 | 2247 - 2252 / 2190 - 2201 |
| 100 | 27.14 | 27.44 | 2355 / 2355 |

With a 20 us gap every slow-half offset is 1.6 us slower and runs about 55 MHz lower; with a
100 us gap the clock is the same everywhere and the halves differ by 0.3 us (1.1%).

## A separate write between GEMMs, C in the fast half

| written after each GEMM | into the fast half (80 MB) | into the slow half (200 MB) |
|---|---|---|
| 64 KB | 28.69 | 28.78 |
| 1 MB | 28.46 | 28.42 |
| 4 MB | 28.89 | 28.74 |
| 16 MB | 29.94 | 29.95 |
| 32 MB | 29.62 | 29.58 |

**Conclusion.**
- **Two parts, one through the clock.** While the GPU runs close to back-to-back (gaps up to 50
  us), C in the slow half costs 0.9-1.7 us per GEMM, and the GPU runs 20-57 MHz lower. At equal
  clocks (100 us gaps, both at 2355 MHz) the halves differ by only 0.2-0.3 us (about 1%). At
  a 20 us gap the slow half is 5.9% slower in time but only 2.5% lower in clock, so it also
  costs cycles (about 3%). The cycle cost shrinks the more idle time follows each GEMM, which
  fits slow-half writes draining more slowly behind the kernel.
- **It is the GEMM's own writes.** Another kernel writing 64 KB to 32 MB into the slow half
  between GEMMs does not slow them any more than the same write into the fast half. Large
  writes (16-32 MB) slow the GEMM by the same amount wherever they go.
