# E4: per-channel and per-XCD L2-to-memory traffic, fast against slow C

`python e4_channels.py`: v9, 4096x4096x1024, bf16, GPU 2. `place.py launch` puts A/B at 900 MB
and C at 16, 48, 80 MB (fast half, E2) and 144, 176, 208 MB (slow half) of one 1 GB buffer, 10
launches each, under `rocprofv3 --pmc` with derived counters
(`reduce(select(EVENT,[DIMENSION_INSTANCE=[i]]),sum)`, definitions in `e4_counters.yaml`) that
pick one of the 16 TCC channels (summed over XCDs) or one of the 8 XCDs (summed over channels).
Median of launches 3-10. Counter collection serialises the kernels, so this compares where the
requests go and how long they take, not kernel time. Full tables: `e4_raw.md`.

| C offset (MB) | half | write requests per channel | write requests per XCD | write latency, channel mean (cycles) | write credit stalls per channel |
|---|---|---|---|---|---|
| 16 | fast | 32.8k (all 16) | 65.5k (all 8) | 519 | 35.3-36.0k |
| 48 | fast | 32.8k | 65.5k | 517 | 34.5-35.6k |
| 80 | fast | 32.8k | 65.5k | 516 | 35.2-35.9k |
| 144 | slow | 32.8k | 65.5k | 520 | 34.9-36.0k |
| 176 | slow | 32.8k | 65.5k | 527 | 35.6-36.5k |
| 208 | slow | 32.8k | 65.5k | 524 | 35.6-36.4k |

Read requests are 24.7k per channel and 49.4k per XCD at every offset. Read latency rises from
506 to 534 cycles in launch order (16 MB first, 208 MB last) whichever half C is in, so that is
drift during the run, not placement.

**Conclusion.** The L2 spreads C's writes perfectly evenly over its 16 channels and 8 XCDs
whichever half C lies in, and write latency and credit stalls barely move (1-2%). The halves are
not an L2 channel effect: the address bit that matters is decoded beyond the L2, on the memory
side (data fabric, MALL, HBM stacks or the dies they hang off), and with serialised launches
its cost hardly shows in these counters.
