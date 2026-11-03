# Counters: why v9full loses three points of the wide sweep

**The kernel now uses v9full** (WORKLOG entry 13): in these tables "v9" is the previous order and "v9full"
is the current one.

rocprofv3 1.3.2, GPU 1, `/tmp/v30order/collect_counters.py` (driver `counter_driver.py`): one process per (point,
order, counter pass), 40 launches over 1 GB of rotating copies after 3 warm-up launches, mean per dispatch. The
three points v9full lost in [tile_order_wide.md](tile_order_wide.md), and two two-tile points it won. Kernel time is
the median under the profiler, from a kernel trace in the last pass. Two runs; run 1, with run 2 in brackets
where it differs noticeably.

| point | order | L2 hit rate | L1-L2 read latency (cycles) | L1-L2 reads (M) | L2 misses (M) | DRAM reads (M) | MfmaUtil | us |
|---|---|---|---|---|---|---|---|---|
| 64x64x64 two-tile, 2 trips, K=8192 (lost, 1.063) | v9 | 75.7% | 403 | 12.88 | 3.18 | 3.07 | 17.0% | 124.1 |
| | v9full | 79.1% | 469 | 12.88 | 2.73 | 2.62 | 15.6% | 133.4 |
| 128x256x64 two-tile, 2 trips, K=8192 (lost, 1.026) | v9 | 74.2% | 318 | 39.17 | 10.48 | 9.60 | 65.0% | 346.7 |
| | v9full | 76.8% | 429 | 39.17 | 9.40 | 8.51 | 63.2% (62.0%) | 356.6 |
| 128x256x64 spread, K=16384 (lost, 1.034) | v9 | 75.3% | 500 (524) | 44.53 | 11.22 | 10.60 | 62.0% | 407.7 |
| | v9full | 75.3% | 539 | 44.53 | 11.24 | 10.62 | 61.0% | 429.6 |
| 128x256x64 two-tile, K=4096 (won, 0.96) | v9 | 69.9% | 439 | 13.62 | 4.43 | 3.68 | 52.9% | 136.0 |
| | v9full | 75.0% | 436 | 13.62 | 3.68 | 2.92 | 54.0% (55.4%) | 130.2 |
| 256x256x64 two-tile, 2 trips, K=8192 (won, 0.98) | v9 | 70.7% | 352 | 52.45 | 16.20 | 14.40 | 79.6% | 595.6 |
| | v9full | 72.9% | 319 | 52.45 | 14.97 | 13.17 | 80.3% | 575.6 |

L2 pipeline and memory queue (one run):

| point | order | L2 tag stall / request | L2 busy (Mcycles) | EA read queue level (G) |
|---|---|---|---|---|
| 64x64x64 two-tile, 2 trips | v9 / v9full | 0.006 / 0.006 | 31.4 / 37.1 | 2.31 / 2.30 |
| 128x256x64 two-tile, 2 trips | v9 / v9full | 0.012 / 0.012 | 73.0 / 77.9 | 5.89 / 5.93 |
| 128x256x64 spread, K=16384 | v9 / v9full | 0.013 / 0.015 | 92.2 / 93.6 | 8.01 / 7.96 |
| 128x256x64 two-tile, K=4096 | v9 / v9full | 0.034 / 0.030 | 31.7 / 29.5 | 2.86 / 2.29 |
| 256x256x64 two-tile, 2 trips | v9 / v9full | 0.014 / 0.016 | 119.0 / 118.8 | 9.23 / 8.03 |

## Reading

- Where v9full loses, the L1-to-L2 read latency rises (+16% at 64x64x64, +35% at 128x256x64 two-tile, +3-8% on
  spread) and MFMA utilisation falls with it, even though v9full's L2 hit rate is as high or higher and it misses
  less. Where it wins, the hit rate rises and the latency stays level or falls.
- So the loss is latency, as the TCP-window hypothesis says, but not from more L2 misses. The L2 tag pipeline does
  not stall more, and the memory-side queue is no deeper.
- **The LLC, not the L2** (follow-up below): the L2 counters count every L2 miss as a memory request
  (`TCC_EA0_RDREQ_DRAM` equals `TCC_EA0_RDREQ`), whether the 256 MB LLC serves it or HBM does, so they cannot
  separate the two. With the inputs kept warm in the LLC the two-tile losses disappear, so the extra latency is
  misses going to HBM rather than late hits in the L2, which is cold at every launch either way.

## Follow-up: LLC-cold against LLC-warm inputs

`/tmp/v30order/warm_vs_cold.py`, two runs. Cold: 1 GiB of rotating copies, as in every other timing here. Warm: the
same copy every launch.

| point | cold: v9full / v9 | warm: v9full / v9 | v9 us cold, warm | A+B per persistent wave, v9 / v9full |
|---|---|---|---|---|
| 64x64x64 two-tile, 2 trips (3136x1024x8192, 71 MiB) | 1.062-1.063 | 0.991-1.000 | 121, 91 | 62 / 40 MiB |
| 128x256x64 two-tile, 2 trips (6272x4096x8192, 211 MiB) | 1.030-1.031 | 0.998-0.999 | 326, 310 | 160 / 96 MiB |
| 128x256x64 spread, K=16384 (2048x7168x16384, 316 MiB) | 1.026-1.029 | 1.015 | 394, 385 | 192 / 288 MiB |
| 128x256x64 two-tile, 1 trip (4224x4096x4096, 98 MiB) | 0.957-0.959 | 0.991 | 128, 115 | 56 / 48 MiB |
| 256x256x64 two-tile, 2 trips (12544x4096x8192, 358 MiB) | 0.980-0.981 | 0.980 | 560, 556 | 248 / 160 MiB |

- The two two-tile losses are HBM effects of LLC-cold inputs: warm, v9full ties v9. Their persistent waves touch
  less A + B under v9full, so it is not LLC capacity. It is when and how the first touches go to HBM, which the
  per-XCD footprint model does not see.
- Spread at K = 16384 keeps half its loss when warm. There, v9full's wave touches 288 MiB, more than the LLC holds,
  against 192 MiB for v9, so capacity misses remain.
- The one-trip two-tile win also shrinks when warm (0.957 to 0.991): part of v9full's gains are HBM effects too.
  256x256x64, whose problem does not fit in the LLC, does not change.
- Every other timing in these results uses cold inputs. That is the conservative case for weights; inputs that stay
  LLC-resident between launches would see smaller differences between the orders.

## HBM traffic model

`/tmp/v30order/llc_model.py`: a 256 MiB LRU LLC over (A or B strip, K-pair) chunks. Each persistent wave runs its
tiles in lockstep over K (last pair first, then 0, 1, ...), and the Stream-K tiles run as one final phase. Cold
start; writes and the L2 are ignored. Compulsory traffic is all of A and B read once.

| point | A + B (MiB) | v9 HBM (MiB) | v9full HBM (MiB) | v9full / v9 | measured v9full / v9 (cold) |
|---|---|---|---|---|---|
| 64x64x64 two-tile, 2 trips | 65 | 65 | 65 | 1.000 | 1.063 |
| 128x256x64 two-tile, 2 trips | 162 | 162 | 162 | 1.000 | 1.026-1.031 |
| 128x256x64 spread, K=16384 | 288 | 288 | 398 | 1.38 | 1.026-1.035 |
| 256x256x64 spread, K=16384 | 352 | 352 | 608 | 1.73 | 1.008 |
| 128x256x64 two-tile, 1 trip | 65 | 65 | 65 | 1.000 | 0.957 |
| 256x256x64 two-tile, 2 trips | 260 | 276 | 262 | 0.95 | 0.98 |
| 256x256x64 one-tile S=16, K=8192 | 132 | 132 | 132 | 1.000 | 0.995 |
| 128x256x64 spread, K=4096 | 72 | 72 | 72 | 1.000 | 0.973 |

- Spread at large K: v9full's persistent wave spans more B columns than the LLC holds across K, so it re-reads
  1.38-1.73x the compulsory bytes where v9 reads them once. That is the spread loss, which is why half of it survives
  warm. At 256x256x64 the extra traffic is still far below HBM bandwidth (about 1 TB/s) and costs 0.8%.
- 256x256x64 two-tile, 2 trips: v9's waves spill slightly (1.06x), v9full's do not, matching v9full's 2% win.
- The two two-tile losses read exactly the compulsory bytes in both orders, since everything fits in the LLC. The
  difference is how those bytes are requested over time, not how many. Untested: in its first wave v9 touches
  more distinct strips at once (46 A + 16 B per K-step, against 32 A + 8 B), so more unique HBM requests are in
  flight, while v9full's shared lines leave more CUs waiting on the same fetches.
