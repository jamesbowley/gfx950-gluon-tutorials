# v30_streamk_streamlined work log

V30 starts from v29's proven quadrant path and makes its production contract
explicit. The top-level device kernel is new; stable quadrant helpers are
imported from v29 so this change does not duplicate or perturb their pipelines
and synchronization protocols.

## 1. Scope and structure (2026-10-22)

Files:

- `selection.py`: exact flat config validation and mechanical launch constants.
- `heuristics.py`: current auto-tile and capped rules.
- `runtime.py`: partial workspace, owner flags, barriers, and L2 flags
  (L2 epochs until entry 5).
- `matmul_kernel.py`: the explicit quadrant-only launcher and device kernel.
- `check_streamk.py` / `run_checks.sh`: correctness, guard, re-arming, and
  stale-barrier checks.

The supported tiles have BM and BN in `{64, 128, 256}` and BK in `{64, 128}`.
Every build uses four waves and 256 programs. Small tiles, transpose/swap,
small-tile residency checks, and fallback dispatch are deliberately absent.

The host contract follows AITER: row-major/K-contiguous A and `(N,K)` weights,
plus row-major C, aligned padded leading strides, nonnegative strides, and
signed 32-bit tile-local byte offsets. The launcher gives the device kernel a
zero-copy `(K,N)` view of the weights.

## 2. Selection and split simplification

`matmul` accepts AITER's flat config dictionary and requires every tunable
value explicitly; it reads no `STREAMK_*` environment variables. The current
tile and capped thresholds live in `heuristics.py` only for offline config
generation. `selection.py` mechanically validates the resulting flat,
JSON-safe configuration and mirrors the launcher's primitive shape/topology
calculations for offline pruning. The launcher parses and calculates directly;
neither path calls the heuristic module.

`GROUP_SIZE_M` is a config value. The runtime policy must be `dp`, `one_tile`,
`two_tile`, or `spread`, and the fixup must already be `owner` or `rs`.
`capped`, `auto`, omitted tiles, and an omitted one-tile split are rejected.
CU and XCD counts come from AITER-compatible host helpers.

Tile-aligned Stream-K is uniform:

```text
working programs = S * split_n
```

`split_n` is validated against `256 // S` and pairs per tile. The launcher
derives the working count, compact grid, reader count, owner peers, and L2 XCD
split.
`N_HI`, `EXTRA`, `RS_R_HI`, and a raw active-program override are gone.

This deliberately differs from v29 when its capped rule reached the floor
`256 // S` and then used all 256 programs despite a remainder. V30 leaves that
remainder idle. For example, at 65 tiles and 32 K-steps, v29's capped result is
one-tile owner on all 256 programs (61 tiles get four programs and four get
three); v30 uses a uniform three-way split on 195 programs. The already-dropped
`S > P/2` one-tile uneven policy remains
replaced by DP or spread. Two-tile and spread use the full 256-program grid and
retain their end-to-end `L/L+1` pair-range balance.

## 3. Paths retained

- Persistent cross-tile prefetch and the four-quadrant paired K loop.
- K_PAD and the masked-physical-tail-first rotation.
- Owner and reduce-scatter one-tile endings.
- WT and XCD-local L2 partial exchange.
- Reversed two-tile and spread schedules.
- Ragged M/N/K, bias, and compact no-full-tile launch.

`STREAMK_FORCE_EDGE`, `LAST_LAUNCH`, global `WARMUP`, and `_RESIDENCY` are not
part of v30 production code.

## 4. Checks

The checker ran DP, capped, owner, reduce-scatter, L2, two-tile, spread,
compact, and ragged+bias configurations on:

- `64x64x64`
- `128x128x128`
- `256x128x64`
- `128x256x64`
- `256x256x64`

All 45 configurations matched torch, left output guards intact, and returned
owner flags and barrier counts to zero. The 64x64 set was also run with three
back-to-back launches per configuration, the 128x128 set with two, and the
other focused build checks once.

The stale-count negative control presets a reduce-scatter barrier to an
impossible count. It timed out after 10 seconds as required.

Against v29 on `64x64x64`, shape `1088x1024x512`, uniform four-way one-tile:

- Owner and reduce-scatter outputs were bitwise equal.
- Both versions used 35,712 bytes of shared memory.
- Generated AMDGCN text was smaller in v30:
  - owner: 185,607 to 179,713 bytes;
  - reduce-scatter: 167,837 to 163,399 bytes.

On a `256x256x64` DP build, both versions used 151,424 bytes of shared memory
and produced the same result; AMDGCN text size was 253,969 versus 253,904
bytes. Text is not expected to be byte-identical because the top-level kernel
signature and dispatch branches changed.

## 5. Graph-safe L2 flags (2026-10-22)

**The bug.** The `l2` barrier stored a host-side epoch, passed as a kernel
argument, and waited for every peer's flag to equal it. A CUDA graph freezes
that argument, so replaying a captured launch reuses its epoch, and its flags
already hold that value from the previous replay. v29's "safe under CUDA
graphs" note held only for its timing harness, where each graph cycles
through 100 different launches and consecutive launches always differ.
`check_streamk.py --graph` on the epoch code: capture alone left the flags
armed (`graph capture left L2 flags armed`). Poisoned partials did not show
wrong results at 64x64x64 because every program reached the barrier at about
the same time; the armed flags are the deterministic symptom.

**Per-reader flags, re-armed in the launch.** No atomics, no epoch
(`l2_tile_barrier`).
- The tile whose first slot is `b` owns `n * PER` words from
  `tile_sync[2P + b * PER]` (`PER = P / XCDs = 32`). Reader `i`'s row is the
  `n` words at offset `i * n`; writer `j` owns word `j` of every row.
- After its drain, writer `j` stores 1 to word `j` of all `r` rows: one
  masked store.
- Reader `i` polls its row with one `.cg` load until all `n` words are 1,
  then zeroes the row. It is the row's only consumer, so the reset cannot
  race with another program, and stream order puts the next launch after it.
- The flag region is zero between launches. `tile_sync` grows to
  `2P + P * PER` words (33 KiB); `next_epoch` and the epoch argument are gone.
- `spid // PER` fixes each slot's XCD, so a tile's region is only ever
  touched from one XCD; the wt path never touches it.

**Two problems found on the way.**
1. **Waves of one program.** The first version used v29's 64-element flag
   tensor, which every wave holds a copy of, so every wave stored, polled and
   reset on its own. Wave 0 of a reader could zero the row while waves 1-3
   still polled it, and they spun forever: 1088x1024x512, 64x64x64, split 4
   (S = 16, one pair per program, after a full-tile wave) hung every time.
   The epoch never went back to an old value, so v29 never had this. A
   lagging wave's duplicate 1 could also land after the reset and arm the
   next launch.
   - Fix: only wave 0 stores flags and resets (elements `64w .. 64w + 63`
     of a `64 * num_warps` tensor belong to wave `w`; the flags are in
     `m < 64`). Every wave still polls its own copy, and an `s_barrier`
     after the poll keeps the reset behind every wave's last poll.
   - Without that barrier the same shape hangs again (an A/B run on the
     anchor point, killed).
   - Making the poll's `min` a workgroup-wide reduction instead (all waves
     exit together) was also correct, but no faster.
2. **Cache lines.** The first layout gave each reader its own 128-byte row
   (`(b + i) * PER`). At n = 4 a writer then touched 4 lines and the readers
   polled 4, where v29's `n` flags shared one: the 128x128 anchor ran
   0.42 us (1.5%) slower than v29 `l2`. Packing the tile's rows
   (`b * PER + i * n + j`, line-aligned since `PER` words are 128 bytes)
   puts the whole grid in one line up to n = 7 and brought the anchor back
   to level (below).

**Checks** (`run_checks.sh`, [results/checks.txt](results/checks.txt)):
- All 45 configurations on the five tiles pass with three launches each.
  `synchronization_rearmed` now also requires the whole flag region to be
  zero.
- `--graph`, all five tiles: 20 rounds each of a single-node graph, a
  two-node graph (two L2 shapes: one tile at the largest XCD-local split,
  the 17-wave tail at split 4), and an eager launch followed by a replay.
  Fresh inputs every round, the partial workspace poisoned with NaN before
  each, results and the re-armed state checked after each. All pass.
- `--stale-flag` (negative control): every flag preset to 1, partials
  poisoned, three pairs over two programs so reader 1 reaches its slice a
  pair before chunk 0 stores. It gives a wrong result, as required, at
  128x128x128 (5 of 5 runs). At 64x64x64 and 256x256x64 it does not: the
  skew is shorter than the reader's own store and drain there. The script
  runs it at 128x128x128.
- `--stale-count` still hangs.
- `test_selection.py`: 19 pass.

**Assembly** ([asm_check.py](asm_check.py),
[results/asm_check.md](results/asm_check.md)): one-tile rs, split 8 on 17x16
tiles. The `l2` build has no atomics, no `buffer_wbl2` and no `buffer_inv`;
its poll and slice loads are `sc0 nt` (17 at 256x256x64 and 128x128x128, 9
at 64x64x64). VGPRs equal wt's (488 / 256 / 84), no spills.

**Timing** ([timing.py](timing.py), [results/timing.md](results/timing.md),
GPU 1): v29 timing.py's method, CUDA graphs of 100 launches over 2 GB of
rotating inputs, 9 shuffled rounds of 5 replays, median us per launch. v29's
epoch `l2` is correct in this harness and is the reference. Every variant
runs the split v29's XCD-local rule resolves; v29 wt is given the same
program count.

| point | tile | shape | S | n | v29 l2 | v30 l2 | v29 wt | v30 wt | v30 l2 / v29 l2 | v30 l2 / v30 wt |
|---|---|---|---|---|---|---|---|---|---|---|
| anchor rs | 128x128x128 | 1024x1024x7168 | 64 | 4 | 28.06 | 28.16 | 27.26 | 27.16 | 1.003 | 1.037 |
| S = 8 | 256x256x64 | 3072x5632x8192 | 8 | 32 | 210.79 | 210.90 | 210.70 | 210.16 | 1.001 | 1.004 |
| S = 16 | 256x256x64 | 4352x4096x8192 | 16 | 16 | 217.74 | 218.32 | 217.91 | 218.32 | 1.003 | 1.000 |
| S = 32 | 256x256x64 | 4096x4608x8192 | 32 | 8 | 225.24 | 224.58 | 224.81 | 224.90 | 0.997 | 0.999 |

- **The new `l2` is level with the epoch version**: 0.997-1.003, against a
  0.4-2.3% spread of each point's own rounds.
- **v30 wt is level with v29 wt** (0.996-1.002), so the kernel around the
  barrier did not change speed either.
- **`l2` does not beat wt on these points.** It is 3.7% slower on the
  anchor, as v29 found (27.23 against 27.89 us), and level on the 256x256
  points. The reasons for `l2` remain those of v21: a shorter barrier in the
  ending, which these whole-kernel times do not show.

**Follow-up: `gl.barrier()` and the two compiler fences (2026-10-23).**
- The barrier before the reset is now `gl.barrier()` instead of inline-asm
  `s_barrier`. It lowers to `s_waitcnt lgkmcnt(0)` then `s_barrier`: no
  vector-memory wait and no cache fences. The hang shape, `--graph` on three
  tiles, `--stale-flag` and the assembly check pass. Timing is level with
  v29's epoch `l2` (1.000-1.003; [results/timing.md](results/timing.md)).
- **The fence inside the poll loop is required.** Built without it (compiled
  only, never launched), the flag load and the whole poll loop are gone: the
  128x128x128 `l2` build has 16 `sc0 nt` loads instead of 17, and no
  backward branch compares the flags. The loop body loads an address nothing
  in the loop writes, so in the compiler's single-thread model the load is
  loop-invariant, and a loop that then only spins can be assumed to end and
  deleted. The barrier would vanish and readers would sum partials early,
  wrong results rather than a hang. The asm memory clobber makes the load
  look possibly changed on every iteration.
- **The fence after the wait is not needed by today's compiler.** Without
  it, the 128x128x128 and 256x256x64 builds lose only the fence's own
  instructions; the 64x64x64 build is laid out differently, but the first
  partial load still follows the barrier in all three. It stays: nothing
  else orders the slice loads after the poll in the compiler's model
  (`s_barrier` does not count as a memory access to it, and the reset store
  does not alias the partials), so a later compiler could hoist them.

## 6. Load offsets and store hints (2026-10-23)

Two changes, each kept only because no measured shape got slower. Timing is
`triton.testing.do_bench` medians of 7 interleaved base/new rounds on an idle
GPU; a base-against-base run put the noise at about 1%.

**One offset tensor per operand on full tiles.** Full tiles used to keep a
second offset tensor for the odd K-step (`*_offsets_next`) while ragged tiles
already shifted the scalar base by `odd`. Full tiles now do the same, so an
operand's offsets are `(top, bot, half, odd)`. The hot loop gains 24-32 scalar
adds (the 64-bit base arithmetic for `+ odd`); VGPRs fall 12 at 128x128x128,
rise 8-12 at 128x256x64 and are unchanged at 256x256x64, with no new spills.
All 15 timing shapes were within noise (0.98-1.015; the borderline 64x64x64
dp and S=16 points went both ways on repeat runs).

**v9's store-side `multiple_of` hints.** The C row term at the four store
sites and `n_left` carry `multiple_of(..., 8)`, and the wrapper now rejects
`N % 8 != 0`, which the `n_left` hint relies on. Only builds whose `stride_cm`
is not a multiple of 16 change: the 64x64x64 ragged build (N = 1016) goes from
36 sub-dword stores to 16-byte stores and from 328 to 88 VGPRs. N = 4104 is
9% faster at 256x256x64 and 2.6% at 128x128x128; every other shape is within
0.2%. Edge guards hold for N = 4104 at all three tile sizes.

## 7. Separate EVEN_M and EVEN_N (2026-10-23)

The launcher now sets `EVEN_M = M % BLOCK_M == 0` and `EVEN_N = N % BLOCK_N == 0`
independently; before, a ragged M or N set both. This re-tests v25's finding
([v25 WORKLOG](../v25_streamk_ragged_mn/WORKLOG.md), "One flag alone spilled"):
with the old offsets a mixed build held one operand as `offsets` + `offsets_next`
and the other as `top` + `bot` with a scalar `odd`. Since section 6 both forms
use a scalar `odd`, and the mixed builds no longer spill more.

- **Correctness.** `check_streamk.py` gains `ragged_m` and `ragged_n` cases; all
  pass on 64x64x64 and 128x128x128. 120 M-only / N-only builds (five tiles; dp,
  owner, rs, l2, two-tile, spread; with and without bias) are correct with the
  sentinel guards intact.
- **Registers.** The hot loop's VALU count falls in every mixed build (for
  example 166 to 108 at 128x128x128 M-only dp) and VGPRs mostly fall. Spills
  change only at 256x256x64: ten builds spill less (N-only two-tile with bias
  47 to 20, spread with bias 56 to 26, N-only owner/rs with bias to 0), one
  spills more: M-only two-tile without bias, 15 to 19.
- **Timing** (GPU 5, interleaved, base-against-base within 0.2%): 4000x4096x8192
  at 128x128x128 is 3.3% faster on three runs. Everything else is within 1%:
  the M-only two-tile build that gained spills measured 0.996-1.011; M-only
  4000x8192x8192 at 256x256x64 and 256x128x64 were consistently 0.4-0.7%
  slower. Aligned and doubly ragged controls are unchanged.
