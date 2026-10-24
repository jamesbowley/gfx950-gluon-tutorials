# Load round trips on MI355X: L2, memory-side cache (LLC) and HBM

[docs/tcp_bound_tile_model.md](../../docs/tcp_bound_tile_model.md) (branch `tcp-bound-tile-model`) caps a CU's loads at C / L, with C = 44 KB (TCP plus queue) and L = 1000 cycles. This experiment measures what L is, which L goes with C, and whether a GPU-clock change moves it in cycles or in ns.

All runs are on physical GPU 0 (MI355X, bus 0000:05:00.0), fp16/bf16, with the shader clock unpinned. Except where stated, the clock was 2.36-2.40 GHz, read from each ATT trace and from the kernel's own counters. fclk was 1250 MHz and mclk 2000 MHz throughout.

## Summary

| | value at ~2.4 GHz | how solid |
|---|---|---|
| HBM round trip, unloaded (4 loads in flight per CU) | 964-984 cycles, about 400-410 ns | repeated, three methods agree |
| LLC round trip, unloaded | 542-656 cycles | repeated; varies more between runs |
| L2 round trip, unloaded | 200-208 cycles | repeated |
| loads a CU can have in flight | 44-48 | measured directly: issue stalls start there |
| one CU alone: HBM / LLC / L2 bandwidth | ~26 / ~46 / 64 B/cycle | direct; 64 is the TCP's processing rate |
| all 256 CUs: HBM / LLC / L2 bandwidth per CU | ~11-12 / ~14 / ~60 B/cycle (~7.0 / ~8.5 / ~37 TB/s) | direct |
| round trip with a full window | 44 KB / the bandwidth above: ~1700 cycles (one CU, HBM), ~3700-4400 (all CUs, HBM) | direct |
| an L2 or LLC hit queued among HBM misses | takes as long as the misses (1044 against 1064 cycles) | direct, per load |
| cycles or ns | **not determined** | the clock could only be moved 10% (below) |

What this means for the doc:

- **1000 cycles is the unloaded HBM latency at 2.4 GHz.** That number is real. It is not the L that goes with C = 44 KB.
- **With the window full, the round trip is set by bandwidth.** It is 44 KB divided by what the CU gets, which depends on the hit mix and on how many CUs are loading. L is therefore not an independent constant, and the tile test is better written in bandwidth terms (section 5).

## 1. Method: one load's round trip, directly

`depth_kernel` in [mre.py](mre.py):

- One workgroup per CU, 4 waves. Each load is a `buffer_load_dwordx4 ... lds`: 1 KB per wave, 4 KB per workgroup.
- After every load, the wave runs `s_waitcnt vmcnt(D)`. It therefore holds at most D + 1 loads, and the wait after load i ends when load i - D returns. In the ATT trace, that wait's end minus load i - D's departure is that load's round trip from issue. A wait that stalled is exact; otherwise its end is only an upper bound.
- Sweeping D from 0 to 16 takes the CU from 4 to 68 loads in flight, past the window.
- Tiers:
  - **L2:** a 512 KB per-XCD region, primed in the same kernel.
  - **LLC:** a region re-read after 16 MB per XCD of other reads have pushed it out of L2.
  - **HBM:** addresses nothing else touches, after a 1 GiB flush.
  - **Mixes:** load j is a hit when j % 4 < h4.

Validation, before the sweeps:

- The burst loop has 16 loads, 16 `s_waitcnt vmcnt(D)` and no `s_barrier` (`results/clock_unpinned/depth_*.s`).
- L2 counters on the burst alone: L2 tier 100% hits; LLC and HBM tiers 0% hits, with memory bytes equal to the burst (1.05 MB on one CU, 268 MB on all CUs).
- At D = 0, the tiers match the earlier single-load probe (`experiments/tcp_latency`: 226 / 558 / 985 cycles) and this experiment's race control (L1 208, L2 296, HBM 1220 cycles, the last two including ~75 cycles of timer overhead).
- At small D, loads in flight = 4(D + 1), and rate x mean round trip equals loads in flight (Little's law).

## 2. Results (unpinned, `results/clock_unpinned/depth_sweep.json`)

Round trip from issue (median of exact samples, cycles), and the CU's rate:

| loads in flight per CU | 4 | 11 | 19 | 27 | 35 | 43 | 46 | 49 |
|---|---|---|---|---|---|---|---|---|
| one CU, HBM | 964, 4.2 B/c | 1024, 11.3 | 1102, 17.8 | 1192, 24.0 | 1444, 25.7 | 1692, 25.9 | 1812, 25.7 | 1840, 25.8 |
| one CU, LLC | 578, 7.1 | 652, 18.7 | 668, 29.8 | 680, 40.4 | 752, 46.7 | 940, 46.5 | 992, 45.8 | 1024, 46.2 |
| one CU, L2 | 208, 17.3 | 212, 49.5 | 280, 63.9 | 400, 63.9 | 64.0 (TCP-rate bound) | | | |
| all CUs, HBM | 984, 4.0 | 1368, 8.8 | 1740, 11.4 | 2534, 11.3 | 2992, 12.1 | 4360, 10.7 | 3952, 12.0 | 4320, 10.3 |
| all CUs, LLC | 656, 6.6 | 888, 13.3 | 1296, 15.4 | 2110, 13.0 | 2328, 15.2 | 3288, 13.9 | 3436, 13.5 | 3756, 14.3 |
| all CUs, L2 | 200, 17.6 | 224, 46.9 | 328, 54.7 | 464, 56.8 | 556, 61.0 | 762, 57.6 | 744, 61.4 | 820, 61.0 |

- **The window is 44-48 loads.** Issue stalls are 0-1% with 43 in flight, 4-61% at 46, and 76-99% at 49. One CU reading L2 also stalls at issue from ~33 loads in flight, but that is its 64 B/cycle TCP rate, not the window.
- **Once a tier's bandwidth limit is reached, more loads in flight only add round trip** (Little's law). One CU from HBM levels off at ~26 B/cycle from ~27 loads in flight on, and every further load just waits longer.
- **Hits wait behind misses**, measured per load. With half L2 hits and half HBM misses, hits take 1044 cycles and misses 1064 (one CU, 19 in flight); on all CUs it's 1318 against 1376. With LLC hits it's 860 against 884. A hit's slot is held until the misses ahead of it complete.

### Second request for a line still in flight (race kernel, in-kernel timers)

Two requesters read the same cold 1 KB lines, the second a set delay after the first (`results/race_sweep.json`, `results/race.png`):

- **Same XCD, other CU:** the second request takes the first's round trip minus the delay, down to an L2 hit. It adds no memory reads, and L2 counts it as a hit.
- **Same CU:** the same, ending in an L1 hit.
- **Other XCD:** its L2 reads the line again from the memory side, at ~830 cycles.

### GEMM tile pattern (256 x 256 tiles, loads only, 256 workgroups)

- L2 hit 72% (counters).
- Both traced CUs keep 82-86% of loads stalled at issue the whole run, so their windows are full. They move 30-33 B/cycle with cold inputs, and 41-44 B/cycle with inputs written just before the kernel (inputs then served from the memory side, not HBM).
- In-process times: cold 50.0 µs, inputs warm in the LLC 36.7 µs, freshly written 38.4 µs (K = 2048).

## 3. v7 and v8: where the doc's 1000 cycles comes from

The v8 README's "~1000 cycles" (§4.5) is how long **v7's** schedule takes from its first load to a full TCP and queue. It is set by the MFMAs in between. The README reasons that v7 doesn't stall at K = 8192, so the round trip is under ~1000 cycles, and stalls at K = 16384, so it's over. That is a bound, not a measured round trip.

Re-traced here (`v7_compare.py`, `v8_compare.sh`; `results/v7/`, `results/v8/`):

- **v7 at steady state** (dispatch 600, 1.61-1.76 GHz):

  | K | inputs | loop MFMA efficiency |
  |---|---|---:|
  | 8192 | hot | 97.4% |
  | 16384 | hot | 92.1% |
  | 8192 | cold | 92.0% |
  | 16384 | cold | 78.5% |

  The stalls are at issue (window full); no `vmcnt` wait ever stalls. So the README's no-stall / stall boundary reproduces with hot inputs; with cold inputs v7 already stalls at K = 8192.
- **At dispatch 15**, early in the run, the clock is in a power-management trough (1.24-1.36 GHz, also without ATT), and v7 barely stalls anywhere.
- **v8 never fills its window** (3.6-5.6% of loads stall at issue) and never waits on `vmcnt`, so it can't measure a round trip. Hot vs cold changes its kernel time by 2%.
- **Clocks:** the GEMMs run at 1.2-1.9 GHz, the load-only kernels at ~2.4 GHz. Each XCD's clock follows mainly its own load.

## 4. Cycles or ns: what was tried

- **`rocm-smi --setperfdeterminism 2000`:** a soft maximum; the kernels still ran at 2.36-2.39 GHz.
- **`amd-smi set -L sclk max 2000`:** the overdrive table read 2000, but the active DPM level stayed 2400 and the kernels still ran at ~2.3-2.4 GHz. A hard pin through the range commands needs the overdrive interface (with its warning), which was declined. The clocks were reset to defaults afterwards.
- **MFMA co-runners** (MFMA-only workgroups on other CUs, stopping at a deadline; `corun` subcommand, `results/clock_corun/`). They lower the measured CU's clock to 2.27 GHz (co-runners on its own XCD) or 2.15-2.18 GHz (on all other CUs). The counters show they add no memory traffic. Over that 10% range, on one CU:
  - The L2 round trip (204 cycles) and the TCP rate (64 B/cycle) are constant in cycles.
  - The unloaded HBM round trip stays at 964-968 cycles (403 → 443 ns). Fixed in ns would predict ~880 cycles at 2.18 GHz.
  - One CU's HBM ceiling stays at 26.4-26.5 B/cycle.

  These are single samples per clock over a narrow range, so they are suggestive, not conclusive.
  - The case a GEMM is in, all CUs bandwidth-bound on HBM, was not tested at a lowered clock. HBM delivers a fixed rate in bytes per second, so there the per-CU share, and with it the full-window round trip, is expected to be fixed in ns.
  - A test that could decide it: saturate HBM from every XCD while MFMA co-runners lower only XCD 0's clock.

## 5. The tile test in bandwidth terms

The doc's condition T x C / B >= L (T = MFMA cycles per K tile, B = bytes per K tile) is the same as

```
B / T  <=  C / L  =  the bandwidth one CU actually gets, in B/cycle
```

The left side is the tile's demand; the right side is measured above, so L doesn't need to be known separately:

| regime | per-CU bandwidth (B/cycle at ~2.4 GHz) |
|---|---|
| all loads hit L2 | 64 (the TCP's rate) |
| one CU's misses go to HBM, others idle | ~26 |
| one CU's misses go to the LLC | ~46 |
| all CUs, misses to HBM | ~11-12 / (miss fraction), capped at ~26 |
| all CUs, misses to the LLC | ~14 / (miss fraction), capped at ~46 |
| measured, GEMM pattern, 72% L2 hits | 30-33 (cold), 41-44 (inputs in the LLC) |

- **Hits behind misses.** Any misses in a CU's window cap that CU at its miss ceiling (~26 B/cycle for HBM), whatever the hit fraction, until it is close to 100%.
- **On the full chip, the hits' benefit** is that HBM delivers 1 / (miss fraction) times more useful bytes. The all-CU rows above, and the hit-fraction sweep (`results/clock_unpinned/mix_sweep.json`), follow min(64, 26, 11.3 / miss fraction) within ~10%.
- **Demand per tile** (fp16, 16x16x32 MFMA, 4 waves): 256 x 256 needs 64 KB / 2048 cycles = 32 B/cycle; 256 x 128, 48 KB / 1024 = 48; 128 x 128, 32 KB / 512 = 64. Compare these with the row for the kernel's actual hit mix and the number of CUs loading.
- **Units:** these bandwidths are in cycles at ~2.4 GHz. At a GEMM's 1.6-1.9 GHz, the per-CU HBM share in B/cycle should rise in proportion, if HBM's TB/s is fixed (expected, untested). The one-CU ceilings looked fixed in cycles (section 4).

## Superseded or withdrawn

- **D_44 = depart(n) − depart(n − 44)** (earlier sections of this experiment, `att_latency.py`) assumed a 44-load window and in-order slot release. It equals 44 KB / bandwidth and is replaced by the direct per-load round trip above. Its values are consistent with the direct ones where the window was full.
- **The first hit-fraction run** had an `s_barrier` before every load, and its "cold" data sat in the memory-side cache. Its conclusions were withdrawn. `mix_sweep.json` is the corrected run.
- **The first co-runner run** polled a flag with device-scope atomics, and that traffic slowed the measured CU's memory path (L2 rate 64 → 35 B/cycle). It was replaced by deadline-based co-runners, under which the rate stays at 64.
- **ATT and the profiler:** a launch traced or profiled right before the measured one loses its cache state. Hot inputs warmed by a traced launch looked cold under ATT, while an untraced timing showed them 27% faster.

## Running

```bash
python mre.py --gpu 0 --label unpinned suite          # race control, depth sweep, hit-fraction sweep, GEMM pattern
python mre.py --gpu 0 --label corun corun             # depth sweep at the co-runner clocks
python mre.py --gpu 0 race --sweep                    # second request for a line in flight
python v7_compare.py --gpu 0 run && python v7_compare.py report
GPU=0 bash v8_compare.sh
python clock_check.py --gpu 0                         # is the shader clock where it was set?
```

`mre.py` puts `scripts/triton_deepbind_shim` on `PYTHONPATH` and sets `HIP_VISIBLE_DEVICES` from `--gpu`. Traces are in `tmp/`, reductions in `results/`.
