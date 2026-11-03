# v9_kreverse work log

v9_beyond_hotloop with the K loop run backward on alternate dispatch waves (`K_ORDER`), to test whether a wave can
reuse the k-steps the previous wave just finished, from L2 or the MALL. Everything below is bf16 on physical GPU 1
(amd-smi index 1, BDF 0000:15:00.0; HIP index 3 on this machine, see `env.sh`), the `llir+amdgcnas` stack.

## Verdict (2026-10-28)

- **The L2 idea does not pay.** L2 holds about 10 of a wave's 128-512 k-steps, so a backward wave can reuse very
  little from it. On K = 8192, reversal is within noise of forward or slightly behind (-0.1 to -0.3%). L2 misses don't
  fall; on some shapes they rise.
- **The MALL effect is real, but smaller than modelled.** It appears only when a wave's footprint (about
  `20480 * K` bytes for full-width waves) exceeds the 256 MB MALL. There, odd-wave reversal cut HBM traffic per
  launch as follows:
  - 8192x8192x16384: 1338 to 1022 MB (-24%)
  - 16384x16384x16384: 5302 to 4050 MB (-24%)
  - 8192x8192x32768: 2416 to 2204 MB (-9%)
  - That is about 45-65% of what a strict LRU MALL gives in the model. Reversing every wave (the control) saves
    nothing, so the gain comes from the serpentine sweep, not from walking addresses downward.
- **The speedup is conditional.** Odd-wave reversal changed throughput and energy as follows:
  - 16384x16384x16384: +1.6% TFLOPS in the rocprof trace and +1.1-1.3% sustained, with 1.1-1.8% less energy per
    launch. All 10 of 10 interleaved repetitions were faster.
  - 8192x8192x32768: +0.5% trace (range crosses 0), +0.6% sustained, -0.9% energy.
  - 8192x8192x16384: -0.6% in the trace in both passes, but +0.2% and +0.9% sustained, with up to -1.0% energy.
- **Decision rule:** adopt only if the gain is at least 0.5% and repeatable on large-K shapes, with no regression
  elsewhere. The result does not meet that rule as a default.
  - Only 16384³ passes cleanly.
  - 8192x8192x16384 regresses in the short trace.
  - K = 8192 shapes lose 0.1-0.3%.
- **Status:** v9 stays as is. v9_kreverse is kept as the measured variant, and the open questions are in
  [TODO.md](../TODO.md).
- **Your two concerns, as measured:**
  - Overhead is zero in the hot loop (§2).
  - Latency is not reduced in any way that matters. Mean L2-miss latency drops 5-7% at K = 16384, but the kernel
    hides it. Any gain is energy, and so clock when power-limited.

## 1. Model (before measuring)

Model setup:
- the real `get_pids` mapping (XCD remap, GROUP_SIZE_M = 4);
- an LRU 4 MB L2 per XCD and an LRU 256 MB MALL, tracked in 32 KB chunks;
- dynamic dispatch with ±2% jitter;
- C stores optionally write-allocating.

What a wave reuses:
- Under v9's mapping, wave `w + 1` on an XCD covers the same 4 A row-strips as wave `w`, and the next 8 B
  column-strips.
- So a following wave can reuse A only.
- A k-step on one XCD touches 4 A + 8 B tiles, 384 KB, so L2 holds about 10 k-steps.

Model predictions:
- **L2:** -1 to -4% fewer L2 misses.
- **HBM, forward order:** when a wave's 320 MB (K = 16384) exceeds the MALL, forward order re-reads A from HBM on
  every wave. That is LRU's worst case, a cyclic sweep larger than the cache.
- **HBM, serpentine:** a serpentine sweep keeps about `(256 MB - C_wave) / (a + 2b)` k-steps of A per boundary. Here
  `a` is the A bytes per k-step (1 MB) and `b` the unshared B bytes per k-step (0.25 MB), so about 150-170 k-steps.
- Predicted HBM reads:
  - 8192x8192x16384: -35 to -40%
  - 16384³: -41 to -47%
  - 8192x8192x32768: about -18 to -20%
- **Rotation:** running only the last R k-steps first was worse than full reversal in the model, so it was not
  built.

## 2. Implementation and overhead

The variant and its three orders:
- `K_ORDER = 0`: forward, which is v9.
- `K_ORDER = 1`: programs with `program_id // WAVE_WGS` odd run backward.
- `K_ORDER = 2`: every program runs backward.
- `WAVE_WGS` is the CU count, since one 256x256 program fits per CU.
- Select the order with `KORDER=0|1|2`, or with `matmul(..., k_order=)`.

How backward works:
- It walks the k-step pairs in descending order, `(n-2, n-1), (n-4, n-3), ..., (0, 1)`.
- So v9's `_next` offsets, buffer indices and epilogue are unchanged. Only the pair step changes, to
  `±2 * BLOCK_K`.

Overhead ([results/asm_check.md](results/asm_check.md)):
- `K_ORDER = 0` is v9's instruction stream exactly.
- `K_ORDER = 1` and `2` keep v9's hot loop: 423 instructions, 256 `v_mfma`, 32 `buffer_load`, 64 `ds_read`, no VALU,
  and 456 VGPRs with no spills.
- The only differences are SGPR renaming and the pointer-step operand. It is an SGPR for `K_ORDER = 1` and the
  immediate `-0x100` for `K_ORDER = 2`, against `0x100` in v9.

Correctness ([results/check.md](results/check.md)):
- All six study shapes match torch in every order.
- The backward orders also match v9 bit for bit on these inputs, because random bf16 inputs sum almost exactly in
  fp32.
- An order probe makes the order visible. The k = 0 product is +2^60 and the k = K - 1 product is -2^60. It shows
  that exactly the odd waves (1 and 3 of 4) run backward under `K_ORDER = 1`.

## 3. Measurements

Scripts:
- `sweep.py trace`: rocprofv3 kernel trace, 1000 launches, mean of the last 100. 4 GB of rotating copies, so each
  launch starts with a cold MALL.
- `sweep.py telemetry`: 10 s of back-to-back launches after a 4 s warm-up. It records amd-smi energy, socket power,
  GFX clock and UMC (HBM) activity.
- `sweep.py counters`: rocprofv3 `--pmc`, 20 launches.
- Each repetition runs the three orders back to back, starting from a different order each time.
- Tables: [results/report.md](results/report.md) is the first pass on all six shapes (5 trace repetitions, 3
  telemetry). [results/report_rep.md](results/report_rep.md) is a replication on the large-K shapes plus the
  single-wave control, with two more counter samples.

Noise floor:
- With `K_ORDER = 1`, the single-wave 4096x4096x8192 shape runs every program forward, so it measures nothing but
  noise.
- It read -0.28% and -0.15% in the two trace passes, with per-repetition ranges of about ±0.7%. It read -0.11% and
  +0.04% sustained.

### 3.1 TFLOPS

The table shows the change of `K_ORDER = 1` against `K_ORDER = 0`. Each cell is the mean of the per-repetition
changes; the first value is the first pass and the value after the slash is the replication.

| shape | waves | wave footprint | trace | sustained | energy per launch |
|---|---|---|---|---|---|
| 4096x4096x8192 | 1 | - | -0.28% / -0.15% | +0.04% / -0.11% | +0.20% / +0.51% |
| 8192x8192x8192 | 4 | 160 MB | -0.32% | -0.15% | +0.54% |
| 16384x16384x8192 | 16 | 160 MB | -0.30% | -0.19% | +0.35% |
| 6144x6144x8192 | 2.25 | - | -0.06% | -0.05% | +0.26% |
| 8192x8192x16384 | 4 | 320 MB | -0.60% / -0.66% | +0.20% / +0.89% | -0.05% / -1.04% |
| 16384x16384x16384 | 16 | 320 MB | **+1.64% / +1.57%** | **+1.10% / +1.30%** | -1.10% / -1.77% |
| 8192x8192x32768 | 4 | 640 MB | +0.46% | +0.62% | -0.85% |

`K_ORDER = 2`, which reverses every wave, was -0.5 to +0.2% everywhere and never better than forward beyond noise.

On 8192x8192x16384, the short trace and the sustained run disagree, in both passes.
- The trace is 1000 launches, about 1.4 s. The power controller has not yet capped the clock that early.
- The sustained run is power-limited, at about 1350 W and about 1700 MHz. There, the saved energy buys clock: +2.1%
  and +0.5% GFX clock in the two passes.
- Hypothesis, not yet tested: the gain only shows when power-limited, and a small per-cycle cost shows when not. The
  likely cost is the extra L2 misses in §3.3.

### 3.2 HBM traffic

There is no MALL or HBM counter (`rocprofv3 -L` only exposes TCC/EA counters, which stop at L2). Instead, the bytes
are estimated as `umc_activity × 8 TB/s × time per launch`, using the mean over all telemetry repetitions. For forward
order, this estimate is within about 10% of the model's reads plus C writes on every shape: 145 against 160 MB,
417 against 384, 1395 against 1280, 1338 against 1408, and 5302 against 5632. So it is a usable stand-in.

| shape | forward (MB) | K_ORDER=1 (MB) | K_ORDER=2 (MB) | saved by K_ORDER=1 | modelled saving (strict LRU MALL) |
|---|---|---|---|---|---|
| 4096x4096x8192 | 145 | 144 | 146 | 0 | 0 |
| 8192x8192x8192 | 417 | 406 | 403 | about 0 | 0 |
| 16384x16384x8192 | 1395 | 1403 | 1416 | about 0 | 0 |
| 6144x6144x8192 | 247 | 246 | 249 | 0 | 0 |
| 8192x8192x16384 | 1338 | 1022 | 1304 | 316 (-24%) | 443-508 |
| 16384x16384x16384 | 5302 | 4050 | 5390 | 1252 (-24%) | 2085-2382 |
| 8192x8192x32768 | 2416 | 2204 | 2431 | 212 (-9%) | 450-510 |

The threshold is as modelled: nothing happens while a wave fits the MALL. The saving is about 45-65% of strict LRU.
At 8192x8192x32768, the forward estimate is already 10% under the LRU prediction (2416 against 2688 MB). That
suggests the MALL keeps some data under a cyclic sweep, so its replacement is not strict LRU, which also leaves less
for reversal to win.

### 3.3 L2 counters

There are three samples on the K = 16384 shapes, two on 8192x8192x32768, and one on the others. The table shows the
change of `K_ORDER = 1` against `K_ORDER = 0`.

| shape | TCC_MISS | mean L2-miss latency (RDREQ_LEVEL / RDREQ) |
|---|---|---|
| 4096x4096x8192 | +0.2% | +0.4 to +1.4% |
| 8192x8192x8192 | +0.7% | -2.5% |
| 16384x16384x8192 | +0.5% | -1.0% |
| 8192x8192x16384 | +3.6 to +6.7% | -6.6 to -6.9% |
| 16384x16384x16384 | -0.3 to -3.4% | -5.4 to -6.3% |
| 8192x8192x32768 | +8.9 to +16.3% | -23 to -33% |

What the counters show:
- The modelled 1-4% L2 saving does not appear.
- Reversal raises L2 misses on the 4-wave large-K shapes. `K_ORDER = 2` raises them too, by 1.7-7.5%.
- The mean L2-miss latency drops on exactly the shapes where HBM traffic drops, consistent with more misses served
  by the MALL.
- A guess at why L2 misses rise, not checked: the programs of a backward wave start with fast MALL hits while their
  XCD neighbours are still waiting on HBM. They drift apart in K and share less in L2.

## 4. Open questions

These are listed in [TODO.md](../TODO.md):
1. **Gate by footprint.** Enable odd-wave reversal only when a wave's footprint exceeds the MALL. Then explain
   8192x8192x16384's trace regression: run ATT or `GRBM_GUI_ACTIVE` at a pinned clock to separate the per-cycle cost
   from the power effect.
2. **Persistent kernels (v10+).** Flip direction on alternate tile iterations; the cost is as small there. A
   persistent program's next tile starts right after its last, with no dispatch gap. That may keep an XCD's programs
   in step and avoid the L2 drift in §3.3.
3. **A second GPU.** All of this comes from one GPU.

## Files

- `matmul_kernel.py`: the kernel, with `K_ORDER` and `WAVE_WGS`.
- `env.sh`: the stack and the device. It resolves physical GPU `$SMI_GPU` (default 1) to its HIP index through
  the BDF.
- `check.py`: correctness and the order probe.
- `asm_check.py`: the compiled code against v9.
- `shapes.py`: the study shapes.
- `run.py`: one workload, either N launches or a sustained run with telemetry.
- `sweep.py`: the interleaved trace, counters and telemetry sweeps.
- `report.py`: the markdown tables.
- `results/`: CSVs and reports. `*_rep` and `counters_b` / `counters_c` are the replication.
- Run from `a16w16/`: `source v9_kreverse/env.sh`, then the commands in each script's docstring. The replication was
  run with `--shapes 8192x8192x16384,16384x16384x16384,8192x8192x32768,4096x4096x8192`.
