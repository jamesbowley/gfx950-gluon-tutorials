# Why cache state changes capped vs two-tile

The capped vs two-tile comparison on the 22 close calls
([streamk_capped_vs_auto](../streamk_capped_vs_auto/README.md)) changed with the work run between
launches. Two-tile won after a 100 us sleep and lost after do_bench's 256 MB zero. This experiment
finds out why, by changing one thing at a time and measuring kernel time only.

**Answer.**
- **It is the inputs' cache state and the memory traffic, not the idle gap and not the release.**
  Two-tile sends 11-78% more read requests past L2 than capped on the small shapes, so it gains
  more when A and B are warm in the last-level cache and loses more when they are cold and the
  cache is full of dirty lines.
- **Two-tile's own L2 reuse also depends on the cache state.** On two of the shapes its reads past
  L2 rise by 19-40% when the inputs are cold; capped's are identical in every mode.
- **For the host rule, use a predecessor that writes A** (A warm, B cold), or clean cold caches.
  There, capped and two-tile are level (median +0.6% and +0.2% on the small shapes), which
  supports capped as the default. do_bench's dirty flush biases against two-tile by about 7%, and
  repeated launches on warm inputs bias towards it by about 3%.

```bash
# from the repo root, with the published v10+ stack (see the a16w16 WORKLOGs)
export PYTHONPATH=$PWD/scripts/triton_deepbind_shim:<triton_gfx950-tutorial-v2.2>/python
export LLVM_PASS_PLUGIN_PATH=$PWD/plugins/llir_scheduler/libLlirSched.so TRITON_AMDGCNAS_PLUGIN=1
cd experiments/streamk_cache_state
HIP_VISIBLE_DEVICES=0 python modes.py trace 5632x3072x6144,18432x14592x8192    # kernel-only times
HIP_VISIBLE_DEVICES=0 python modes.py pmc 5632x3072x6144,18432x14592x8192      # counters
HIP_VISIBLE_DEVICES=0 python clocks.py 5632x3072x6144                          # sclk and power
cd ../../kernels/gemm/intra_wave/a16w16 && HIP_VISIBLE_DEVICES=0 python v20_streamk_reduce_scatter/fixup_timing.py \
  --two-tile --shapes 5632x3072x6144 --modes clean_cold dirty_cold clean_warm dirty_warm \
  --launches 10 --jsonl ../../../../experiments/streamk_cache_state/results/release.jsonl
cd ../../../../experiments/streamk_cache_state && python modes.py report        # results/modes.md
```

Full tables are in [results/modes.md](results/modes.md).

## Method

**Shapes.**
- Six small and medium close calls: 5632x3072x6144, 5120x3584x3072, 8192x2304x6144 and
  2048x9216x6144, each with one full wave, and 7680x8960x6144 and 6144x11008x6144, each with four.
- Two big controls: 18432x14592x8192 and 10752x12544x8192.
- Policies: v13 (data-parallel), v20 capped, v20 two_tile. Each shape ran on one GPU, all policies
  and modes in one process.

**Kernel-only timing** (`modes.py trace`).
- Each (policy, mode) runs 10 warm-up and 100 timed launches under `rocprofv3 --kernel-trace`.
- Only the GEMM dispatch's own start and end timestamps count. The gap from the previous
  dispatch's end to the GEMM's start is recorded too: it is how long the GPU sat idle.

**Modes.** Each one is the work run between two GEMMs.

| mode | what the GEMM starts with |
|---|---|
| nothing | the previous GEMM just finished (back to back) |
| zero64 | zero 64 MB |
| sleep | `torch.cuda._sleep`, about 80 us at full clock |
| busy | an ALU-only kernel on every CU for about 400 us (no memory traffic, no idle gap) |
| clean_cold | read 256 MB: caches clean, inputs evicted |
| dirty_cold | zero 256 MB (do_bench): last-level cache full of dirty lines, inputs evicted |
| clean_warm, dirty_warm | the same, then read A and B, so the inputs are back in the caches |
| *_idle | the four above, then the sleep |
| clean_warm_busy | clean_warm, then the busy kernel |
| write_a | read 256 MB, then write A from a copy, as an activation kernel would (A warm and dirty, B cold) |

**Counters** (`modes.py pmc`): 3 warm-up and 10 counted launches per (policy, mode), in three
`rocprofv3 --pmc` passes:
- L2 hit and miss;
- L2 read and write requests to memory, and the subset sent to DRAM;
- L2 lines written back by cache operations (`TCC_ALL_TC_OP_WB_WRITEBACK`, which counts the
  release's `buffer_wbl2`) and by normal evictions;
- DRAM credit stalls;
- GPU active cycles.

gfx950 exposes no last-level cache (MALL) counter. Its hits count as DRAM requests here.

**Release timing** (`fixup_timing.py --two-tile`). `s_memrealtime` stamps time two-tile's phases:
- the head piece's MAC loop;
- the partial-store drain;
- the release atomic (`buffer_wbl2; s_waitcnt vmcnt(0)` before the flag's swap);
- the owner's flag wait and fixup.

These were measured after each cold and warm mode, with the release flag and with a relaxed flag
store, a timing-only control that drops `buffer_wbl2`. The stamps add barriers, so phases run
slightly long. A stamp placed directly after the flag atomic makes the results wrong, so the
release is timed from the drain to the end of the head segment. Every timed launch was correct.

**Clocks** (`clocks.py`). The sysfs sclk and power readings were sampled every millisecond while a
GEMM and one mode's work looped for 2 s.

## Results

### 1. The old "nothing" and "zero 64 MB" columns included host time

On kernels under about 200 us, `between_launches.py`'s CUDA events also time `matmul()`'s host
code: the start event fires on an idle GPU before the kernel is launched. Kernel-only times
against the events (us), on 5120x3584x3072:

| policy | nothing | zero 64 MB | read 256 MB | zero 256 MB |
|---|---|---|---|---|
| v13 | 111.4 (143.2) | 111.5 (138.0) | 113.8 (115.5) | 114.3 (115.4) |
| capped | 100.6 (160.8) | 99.9 (157.8) | 100.4 (102.2) | 102.3 (123.2) |
| two-tile | 95.8 (158.0) | 94.9 (151.4) | 106.0 (108.1) | 112.5 (125.2) |

Reading 256 MB and the sleep agree with the events to within a few percent. Zeroing 256 MB agrees
on the longer kernels, but inflates this one too (capped 102 against 123 us). Kernel times can
differ by up to about 8% between GPUs (6144x11008x6144 ran on another GPU than in the event run),
so compare policies within a shape only.

### 2. Two-tile against capped, by mode

Positive means two-tile is faster. Medians over the 6 small and the 2 big shapes:

| mode | small shapes | big shapes |
|---|---:|---:|
| clean_cold (read 256 MB) | +0.2% | -0.6% |
| write_a (predecessor writes A) | +0.6% | -0.3% |
| dirty_cold (zero 256 MB) | **-6.9%** | -0.2% |
| clean_warm | **+3.4%** | -0.3% |
| dirty_warm | +2.6% | -0.3% |
| sleep (warm, after the previous GEMM) | +3.7% | -0.3% |
| busy | +3.4% | -0.1% |
| nothing (back to back) | +3.0% | +0.0% |

On the big shapes the mode makes no difference between the two policies.

### 3. One factor at a time

Change in kernel time when one thing changes; positive is slower. Medians over the 6 small
shapes, with the range in brackets:

| change | v13 | capped | two-tile |
|---|---|---|---|
| dirty caches, cold inputs | +2.1% | +3.4% (+1.1 to +9.4) | **+11.5%** (+3.3 to +15.6) |
| dirty caches, warm inputs | +2.0% | +1.8% | +2.0% |
| warm inputs, clean caches | -1.0% | -1.2% (-5.3 to +0.5) | **-5.0%** (-12.0 to +0.7) |
| warm inputs, dirty caches | -2.5% | -4.6% | **-13.5%** (-18.2 to -4.0) |
| idle gap: the sleep (four cache states) | -0.1 to +0.0% | -0.3 to +0.4% | -0.8 to +0.5% |
| busy kernel instead of the idle gap | +0.1% | -0.1% | -0.3% |
| A written by a predecessor, B cold (against clean_cold) | -0.6% | -0.7% | -2.3% |

- **The idle gap does nothing.** Hypothesis 3 is ruled out. The old sleep mode's two-tile win
  came from warm inputs: the sleep leaves the previous launch's A and B in the caches.
- **Dirty caches hurt mainly when the inputs are cold.** Then every miss also evicts a dirty line
  to HBM. With warm inputs there are few misses to pay for that.
- **Two-tile reacts 3-4x as strongly as capped** to both warm inputs and a dirty cold cache.

### 4. Why two-tile reacts more: traffic past L2

Read requests from L2 to memory (millions), from the counters:

| shape | capped (every mode) | two-tile clean_cold | two-tile clean_warm | two-tile dirty_cold | two-tile write_a |
|---|---:|---:|---:|---:|---:|
| 5632x3072x6144 | 3.58 | 4.17 | 4.52 | 4.03 | 4.32 |
| 5120x3584x3072 | 2.16 | 2.88 | 2.89 | 2.82 | 2.89 |
| 8192x2304x6144 | 3.63 | 6.46 | 4.90 | 6.86 | 5.47 |
| 2048x9216x6144 | 3.63 | 5.84 | 4.90 | 6.49 | 5.07 |
| 7680x8960x6144 | 12.62 | 14.65 | 13.60 | 14.01 | 14.59 |
| 6144x11008x6144 | 12.03 | 13.33 | 12.07 | 13.68 | 13.44 |

- **Capped and v13 read exactly the same amount in every mode.** The cache state only changes
  what each miss costs.
- **Two-tile always reads more, by 11-78% with clean cold caches**, because its programs start at
  drifting k offsets and share fewer A/B slices. So a warm last-level cache (cheaper misses) helps
  it more, and dirty lines evicted on cold misses hurt it more.
- **On 8192x2304x6144 and 2048x9216x6144, two-tile's reuse itself drops when the inputs are cold**:
  L2 hit rate goes from 64% (warm) to 53-59% (cold or dirty), with 19-40% more reads. My reading
  is that slow memory lets the programs of an XCD drift further apart, so they share fewer
  k-slices. Capped's tile-aligned chunks share them regardless.
- Two-tile also sends 0.45-0.7M more write requests than capped (one 256 KiB partial per program,
  against S x n partials for capped) in every mode.

The counters can't say how many misses hit the last-level cache, so the warm gain on shapes whose
traffic doesn't change (5632x3072x6144, 5120x3584x3072) is attributed to cheaper misses without a
direct count.

### 5. The release is not the cause

Two-tile phases (us) on 8192x2304x6144, one full wave, medians over programs and launches:

| flag | mode | head MAC | store drain | release | owner MAC | owner wait | owner fixup | tail |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| release | clean_cold | 71.0 | 6.1 | 1.56 | 74.2 | 0.6 | 11.6 | 179.3 |
| release | dirty_cold | 83.1 | 5.4 | 1.60 | 76.4 | 0.6 | 13.9 | 202.1 |
| release | clean_warm | 57.7 | 6.0 | 1.32 | 69.9 | 0.7 | 13.8 | 155.3 |
| relaxed | clean_cold | 68.0 | 6.1 | 0.40 | 70.4 | 0.6 | 11.7 | 172.8 |
| relaxed | dirty_cold | 80.8 | 5.6 | 0.36 | 74.2 | 0.6 | 14.4 | 199.1 |

- **The release's `buffer_wbl2` costs 0.6-1.2 us per program (release against relaxed), and a dirty
  cache adds at most 0.16 us to it.** Hypothesis 2 is ruled out.
  - The counters agree: two-tile's releases write back 53-80k L2 lines per kernel against capped's
    8-33k, in every mode.
  - So the write-back is of the GEMM's own lines (most likely C written by other programs on the
    same XCD), not the previous kernel's. That kernel's dirty data is no longer in L2 when the GEMM
    starts.
- **The cache state lands in the head piece's MAC loop**, the first phase of a one-wave kernel,
  when every program streams A and B from memory:
  - dirty cold caches add 5-12 us;
  - warm inputs save 7-13 us.

  The owner's MAC moves much less. On the four-wave shapes the stream-K phases don't react at
  all, because the persistent waves before them have already replaced what the mode left.
- The owner waits under 1 us for its peer's flag in every mode.
- Dropping the release (timing only) shortens the tail by 1.6-6.5 us on 5 of the 6 shapes, and
  lengthens it by 7 us on 7680x8960x6144. That is small next to the cache-state effect on the
  one-wave shapes (11-25 us of tail).

### 6. Clocks and power

- **During the GEMM itself the clock barely moves between modes.** GPU active cycles over
  kernel-only time give 2.1-2.4 GHz per XCD on the small shapes. On three of the four one-wave
  shapes, two-tile ran at a slightly *higher* clock with dirty cold caches than with clean ones,
  while being 12-16% slower.
- **What the loop does between GEMMs sets average power, and so the clock** (sysfs, 2 s loops):

  | loop | small shapes: sclk, power | big shape: sclk, power |
  |---|---|---|
  | sleep, busy, idle modes | 2.16-2.39 GHz, 0.83-1.15 kW | 1.77-1.89 GHz, 1.29-1.40 kW |
  | clean_cold, clean_warm | 2.29-2.36 GHz, 0.90-1.11 kW | 1.83-1.90 GHz, 1.36-1.40 kW |
  | dirty_cold | 2.07-2.14 GHz, 1.03-1.28 kW | 1.75-1.83 GHz, 1.40 kW |
  | nothing (back to back) | 1.99-2.16 GHz, 1.06-1.39 kW | 1.74-1.85 GHz, 1.40 kW |

- Back to back on the small shapes, two-tile draws 1.37-1.39 kW against capped's 1.31-1.33 kW and
  v13's 1.06-1.14 kW.   It runs at the 1.4 kW board limit and at a lower clock. That fits its larger
  back-to-back penalty (sleep to nothing: +3.6% median against capped's +1.3%). Back to back,
  both policies vary widely from launch to launch (p90 16-22% above p10). The big shapes sit at
  the power limit in every mode.

## Recommendation for the host rule

- **Benchmark with a realistic predecessor: `write_a`** (A just written, B cold), or clean cold
  caches (`clean_cold`), which behaves almost the same. Use kernel-only timing.
  - In both, capped and two-tile are level on the small shapes (+0.6% and +0.2% median) and on the
    big ones. That supports capped as the default (v20 WORKLOG entry 7).
- **Treat the other two states as biased, in opposite directions:**
  - do_bench's 256 MB zero penalizes two-tile by about 7% (dirty cold misses);
  - repeated launches on the same inputs (back to back, sleep, cudagraph replays) favour it by
    about 3% (warm misses).

  The real-model sweep (`do_bench_cudagraph`, warm and back to back) therefore leans slightly
  towards two-tile, and it still preferred capped.
- **Where two-tile could win in practice:** a GEMM whose B stays resident in the 256 MB last-level
  cache between calls (small weight matrices reused back to back).

## Next steps

- Two-tile's disadvantage is its traffic past L2, and that traffic grows when memory is slow.
  Aligning its programs' k offsets more tightly would attack both its loss with cold caches and
  its extra power. Its cheaper fixup is already measured: under 1 us of waiting.
- Re-time the host rule's close calls under `write_a` with kernel-only timing, rather than with
  CUDA events or do_bench.
