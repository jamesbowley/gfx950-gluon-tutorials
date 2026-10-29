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

## 8. Peeled first pair in the Stream-K segment (2026-10-27)

The segment's `for i in range(n_pairs)` loop carried `init_acc`'s zero
accumulator in, so every Stream-K build zeroed all accumulator AGPRs once per
segment (`v_mov_b32 v19, 0` and 256 `v_accvgpr_write_b32 aN, v19` at
256x256x64). The first `k_step_pair` is now peeled out of the loop, as in
`persistent_tile`, and its first MFMAs take an inline-0 C operand.

- **Assembly** (the `asm_check.py` builds, l2 and wt alike):

  | tile | zero AGPR writes | inline-0 MFMAs | MFMAs | VGPRs / AGPRs / spills |
  |---|---|---|---|---|
  | 256x256x64 | 256 to 0 | 64 to 128 | 1024 to 1280 | 472 / 256 / 0, unchanged |
  | 128x128x128 | 64 to 0 | 16 to 32 | 512 to 640 | 260 / 80 / 0, unchanged |
  | 64x64x64 | 16 to 0 | 4 to 8 | 64 to 80 | 80 / 20 / 0, unchanged |

  The extra MFMAs are the peeled copy of the `k_step_pair` body.
- **Correctness.** `run_checks.sh`'s commands pass on all five tiles (80
  cases, graph replay), and both negative controls still fail as intended.
- **Timing** (`timing.py`, GPU 1, one run before, two after; v29 is
  unchanged and serves as the reference). v30 / v29:

  | point | pairs per segment | l2 before | l2 after | wt before | wt after |
  |---|---|---|---|---|---|
  | anchor rsall | 7 | 0.928 | 0.928, 0.926 | 0.914 | 0.912, 0.913 |
  | S=8 | 2 | 1.004 | 1.005, 1.005 | 1.005 | 1.002, 1.001 |
  | S=16 | 4 | 1.007 | 1.004, 1.005 | 1.006 | 1.002, 1.006 |
  | S=32 | 8 | 1.007 | 1.009, 1.005 | 1.003 | 1.007, 1.008 |

  All within noise, including S=8, the shortest segments. The zeroing ran
  while the segment's first pair was still loading, so it was most likely
  already hidden.
  The peel stays: it costs no registers and removes the work from short
  segments.

## 9. Last persistent trip prefetches the first Stream-K pair (2026-10-27)

Every persistent trip's epilogue prefetches the next tile's first pair, clamped
to tile `total_tiles - 1`, so each trip issues the same copies and the
`wait_group(5)` counts hold. On the last trip that pair is discarded at the
`wait_group(0)` before the Stream-K phase, whose segment then loads its own first
pair from scratch. For these measurements a temporary switch,
`V30_LAST_PREFETCH` (constexpr `LAST_PREFETCH`, timed by
`timing_last_prefetch.py`), selected the last trip's target. Mode 2 is now the
only path (`SK_PREFETCH`); the switch, mode 1 and the timing script are gone:

- **0, clamp**: the old behaviour.
- **1, mask**: the same copies with every lane masked off (`k_tail_mask` with
  `k_hi = 0`).
- **2, streamk** (kept): the first pair of the program's first Stream-K
  segment (`streamk_first_segment`, computed before the persistent loop). The
  epilogue ends in the state `load_first_pair` leaves (6 groups in flight,
  `a_top` / `b_left` read), so that segment runs with `PREFETCHED` and skips its
  prologue, and the `wait_group(0)` goes. The end-to-end path peels its first
  segment out of the loop for this. Idle programs in a uniform split mask the
  pair off. Without a Stream-K phase, mode 2 compiles clamp's code.

- **Correctness.** `run_checks.sh` passes in all three modes (80 cases each on
  the five tiles, graph replay, both negative controls), and so do
  `check_streamk.py --extra`'s 9 new cases (dp at one and four waves, ragged K,
  rs / l2 tails after three waves, two-tile and spread after a persistent phase)
  on all five tiles. With the target moved one tile row (a negative control),
  every case with a persistent phase and a Stream-K tail fails, and the others
  pass. `asm_check.py`'s synchronization instructions are unchanged.
- **Timing** (`timing_last_prefetch.py`, GPU 1, four runs, the last with mode 2's
  dp fallback; `results/last_prefetch.md`). streamk / clamp and mask / clamp:

  | point | trips | streamk / clamp | mask / clamp |
  |---|---|---|---|
  | S=8, 3072x5632x8192 | 1 | 0.965-0.967 | 0.999-1.000 |
  | S=16, 4352x4096x8192 | 1 | 0.976-0.979 | 0.998-1.002 |
  | S=32, 4096x4608x8192 | 1 | 0.985-0.987 | 1.000-1.002 |
  | S=16, K=2048 | 1 | 0.967-0.969 | 1.003-1.008 |
  | S=16, 3 waves, K=2048 | 3 | 0.978-0.981 | 1.004-1.005 |
  | spread, K=2048 | 1 | 0.979-0.987 | 1.001-1.005 |
  | two-tile, K=2048 | 1 | 0.994-0.997 | 1.005-1.007 |
  | dp, K=1024, 1 and 4 waves | 1, 4 | 1.000-1.003 with the fallback | 1.006-1.011 |
  | dp, K=8192, 1 and 4 waves | 1, 4 | 0.998-1.003 | 0.996-1.003 |

  Spreads are 0.4-2.4%. Masking never pays: the discarded pair costs less than
  the masks (+16-28 VGPRs in the epilogue), so mode 2 falls back to clamp
  without a Stream-K phase rather than masking. The streamk gain (7 us at S=8)
  is larger than one pair's transfer per program. Most likely, without the
  retarget, every program issues its first Stream-K loads at once after the
  `wait_group(0)` and waits on that burst; the retarget overlaps it with the
  last tile's MFMAs and C stores. Not profiled.
- **Registers** (256x256x64). One-tile builds: 472-476 to 480-488 VGPRs, no VGPR
  spills, SGPR spills 0-1 to 26-29, 390-450 fewer instructions (the segment's
  prologue goes). End-to-end builds: the peeled segment adds about 4000
  instructions, and VGPR / SGPR spills go from 7 / 23 to 17 / 46. Spread still
  gains 1.3-2.1% and two-tile 0.3-0.6%. The spill stores run once per program
  and the reloads in each segment's ending (16 to 23 per ending, plus 10 in the
  peeled segment's), none in a K loop. The v26-v29 and committed v30 builds
  already spill 5; entry 8's peel and the `HAS_FULL` change made it 7.
- **Wider sweep** (`timing_last_prefetch.py --set wide --modes clamp streamk`,
  two runs, `results/last_prefetch_wide.md`): 55 points, 11 per tile on all five
  tiles (one-tile rs / owner / l2, capped, two-tile, spread; with bias, ragged
  M / N / K, three waves, K up to 16384). streamk / clamp is 0.974-1.003 and
  agrees between runs within 0.008; the slowest (1.002-1.003) are ragged two-tile
  at 256x256x64 and 256x128x64, within the 0.2-2.4% spreads. Only 256x256x64
  spills VGPRs: the aligned end-to-end builds 7 to 17, but the ragged and bias
  ones fewer than clamp (two-tile ragged 36 to 29, spread ragged with bias 53 to
  47, one-tile rs ragged with bias 23 to 2). The smaller tiles mostly lose
  VGPRs (256x128x64 two-tile 444 to 404) and none spills.

## 10. XCD block helpers, constant split counts, tile order (2026-10-28)

**Code** (no change in the compiled kernels):

- `xcd_blocks` (per, tall) and `xcd_contiguous_id` replace the two copies of the
  XCD remap in `_get_pids` and `get_logical_chiplet_mapped_pids`, and
  `streamk_tile_id` takes its blocks from `xcd_blocks`.
- The uniform non-L2 path passes `SPLIT_N` / `RS_R` as constants rather than
  `spid * 0 + ...`. The reduce-scatter endings build their flag with
  `gl.to_tensor(n_tile > 1)`, so they take a constant or a runtime count (L2).
- A branch-free block start (`x * per - max(x - tall, 0)`) and a restructured
  `streamk_tile_id` compiled to different code on all ten builds compared.
  Keeping the original branch form made all ten (dp aligned and ragged, one-tile
  rs / l2 / ragged l2 / owner / ragged owner with bias, two-tile, ragged spread,
  64x64x64 rs) identical to the previous assembly. A Python check of the
  formulas found no mismatch over 1.2M remap and 10077 `streamk_tile_id` cases,
  the latter against the persistent loop's actual leftover tiles. `run_checks.sh`
  and `--extra` pass on all five tiles.

**Tile order** (`results/tile_order.md`, an experiment; the kernel keeps v9's
order). v9's XCD blocks run over `total_tiles`, so the Stream-K tiles shift
each XCD's block of persistent tiles against the `GROUP_SIZE_M` groups. A
footprint model (distinct A + B strips among an XCD's 32 concurrent tiles)
puts one XCD at 21 strips against 12 on 17x16 tiles. Two variants (scripts in
`/tmp/v30order`, both passing `check_streamk.py` standard and `--extra` on two
tiles):

- **v9full**: blocks over `total_full_tiles`, and the Stream-K tiles last.
- **split**: remapped program id + i P, the order dropped in v15.

Two runs, ratio to v9, agreeing within 0.006 except split at S=32 (1.010, 0.992):

| points | v9full / v9 | split / v9 | model |
|---|---|---|---|
| two-tile, K = 4096-16384, both tiles | 0.981-0.988 | 0.980-0.988 | v9 worse (13 vs 12) |
| one-tile S=16, capped K=4096 / rs K=8192 | 0.980-0.984 / 0.991-0.996 | 0.976-0.979 / 0.989-0.993 | v9 worse (21 vs 12) |
| 129x6 tiles, capped | 0.982-0.984 (256²), 0.965-0.967 (128²) | 0.987-0.988 | v9 worse (15.3 vs 14) |
| one-tile S=8 | 0.996-0.998 | 0.996 | v9 worse (17 vs 16) |
| one-tile S=32 (16x18 tiles) | 1.006-1.011 | 0.992-1.010 | v9 better (12 vs 16) |
| spread K=16384 | 1.015 | 1.013-1.014 | v9 better (Stream-K 10 vs 14) |
| spread K=4096 / 8192, ragged two-tile and spread with bias | 0.998-1.005 | 0.996-1.001 | mixed |
| one-tile rs, ragged with bias (4320x4064x4080) | **1.035-1.037** | **1.031-1.034** | v9 worse (21 vs 12) |
| dp, one and four waves (control) | 0.997-1.003 | 0.997-1.002 | equal |

The model predicts the direction everywhere except the ragged one-tile point
with bias, which is 3.5% slower without v9's order.

**That loss is the row stride and a row-shaped tail** (`results/tile_order_stride.md`):

- Neither the ragged M / N edges nor the bias cause it. Ragged K does, but only
  through the row stride. K = 4080 makes 8160-byte rows, not 128-byte aligned,
  and so does K = 4096 with a stride of 4112: both cost every order 27% (110 to
  139 us), and v9full / split 5% more (1.044-1.049). A ragged K with aligned rows
  (K = 4032, or K = 4080 padded to a 4096 stride) is within 1% for every order.
- The 5% comes from the last S tiles forming one tile row: they share one A strip
  and between them read every B strip. In v9full and split the Stream-K tiles are
  the last S ids, which under `GROUP_SIZE_M` 4 are tile row 16. With
  `GROUP_SIZE_M` 17 the same 16 tiles are one column (1.003), and on 16x17 tiles
  they are a 4x4 block (1.004). The same holds without Stream-K: dp on 17x16
  tiles in split order, where the last partial wave is row 16 on XCD 0, is 15%
  slower with unaligned rows (3.7% aligned), and 0.991 as a column.
- Why a row and not a column, with A and W laid out alike, is not explained; it
  would need cache counters.
- With unaligned rows, two-tile gains more from v9full (0.959-0.960): its 272
  Stream-K tiles span 17 rows.

So v9 stays. An order that places the Stream-K tiles last must also control the
tail's shape (a block or column, not one row), and unaligned rows (any K that is
not a multiple of 64) cost 27% regardless of order.

## 11. v9full on the 55-point sweep: not switched (2026-10-28)

The proposal was to make v9full the only order, for simplicity
(`streamk_tile_id` would become `total_full_tiles + t`), and leave a
configurable XCD mapping for later. It was gated on the 55 wide points of entry
9 (11 per tile on all five tiles): switch if the geomean of v9full / v9 is at
most 1.0 and no aligned-row point is more than 2% slower in both runs
(`results/tile_order_wide.md`, `/tmp/v30order/wide_sweep.py`, two runs).

- **Geomean** 0.9986 / 0.9990 over all 55 points, 0.9977 / 0.9981 over the 40
  with aligned rows.
- **The gate fails** on three aligned points, all of them repeated in both runs:

  | point | v9full / v9 | model (persistent worst XCD, v9 / v9full) |
  |---|---|---|
  | 64x64x64 two-tile, 2 trips, K = 8192 | 1.063-1.065 | 16.5 / 12 |
  | 128x256x64 spread, K = 16384 | 1.034-1.035 | 12 / 12 |
  | 128x256x64 two-tile, 2 trips, K = 8192 | 1.026 | 21 / 12 |

  The footprint model predicts v9full equal or better on all three, so the
  model is missing something here.
- Per point type, the range over the five tiles:
  - Two-tile, one trip: 0.953-0.995, including 0.908 on ragged 128x256x64.
  - Two-tile, two trips: 0.979-1.065.
  - Spread: 0.931-1.037 (best at 128x256x64, worst at 256x128x64).
  - One-tile capped S=16 and l2: 0.983-1.020, slower than v9 except at
    256x256x64. The one-tile S=16 gain of entry 10 holds only for that tile.
  - One-tile owner, three waves: 0.991-1.003.
  - One-tile rs ragged with bias, rows not 128-byte aligned (K = 4080):
    1.029-1.068, the row-shaped tail of entry 10.
- **Coverage**: v9full's persistent tiles plus `total_full_tiles + t` cover
  every tile exactly once in 24288 (tiles, programs, XCDs, policy) cases.

v9full wins most clearly on two-tile with one trip, and loses on a few points
the model cannot explain.

**Counters** (`results/tile_order_counters.md`, rocprofv3). On the three losing
points v9full's L2 hit rate is as high or higher (64x64x64 two-tile: 75.7% to
79.1%) and it misses less. But the L1-to-L2 read latency rises (+16% there, +35%
on 128x256x64 two-tile, +3-8% on spread), and MFMA utilisation falls with it.
On two winning points the hit rate rises and the latency stays level or falls.
L2 tag stalls per request and the memory-side queue are unchanged. So the
losses are latency on tiles close to their outstanding-load limit, not extra
L2 misses.

The L2 counters cannot see the 256 MB LLC: they count every L2 miss as a
memory request. Timed with the inputs warm in the LLC (the same copy every
launch) rather than 1 GiB of rotating copies, the two two-tile losses vanish:
1.062 to 0.991-1.000, and 1.030 to 0.998. Spread at K = 16384 halves, 1.028 to
1.015; there v9full's persistent wave touches 288 MiB of A + B, more than the
LLC, against 192 MiB for v9. So the extra latency is misses going to HBM. On
the two-tile points it is not capacity, since v9full's waves touch less, but
when and how the first touches reach HBM, which the per-XCD footprint model
does not see. Part of v9full's gains are the same effect: the one-trip two-tile
win shrinks from 0.957 to 0.991 when warm.

An LRU model of the LLC (`results/tile_order_counters.md`) gives the HBM bytes
per order. On spread at K = 16384, v9full re-reads 1.38x (128x256x64) and 1.73x
(256x256x64) the compulsory A + B, where v9 reads it once: a capacity effect.
On the two losing two-tile points both orders read exactly the compulsory bytes,
so there the order changes how the same bytes are requested over time.
Possibly fewer unique HBM requests in flight, since v9full's first wave touches
32 A + 8 B strips against 46 + 16, but this is untested. That is too uneven for a default. The kernel keeps
v9's order. The case for a per-shape choice (the planned XCD mapping, with the
Stream-K tiles last and the tail's shape controlled) still stands; it should be
checked against these three points.

## 12. Branch-free XCD remap, renamed round_robin_to_xcd_blocked (2026-10-28)

**Code**: `xcd_contiguous_id` becomes `round_robin_to_xcd_blocked`. The name says
what it maps from (id i placed round-robin, on XCD i % NUM_XCDS) and to (each
XCD's ids one contiguous block). It is a permutation of `[0, count)`, not a
program-to-tile map: `get_logical_chiplet_mapped_pids` renumbers program ids
(count = NUM_PROGRAMS), and `_get_pids` renumbers persistent-loop virtual pids
into tile ids before the `GROUP_SIZE_M` swizzle (count = total tiles). The
per / tall branch becomes the block start the Stream-K pair split already uses:

    xcd * (count // NUM_XCDS) + min(xcd, count % NUM_XCDS) + local

`xcd_blocks` stays for `streamk_tile_id`, which needs both block lengths. A
Python check found it equal to the branch form and a permutation over 72M
(i, count, NUM_XCDS) cases (NUM_XCDS 1-16, count 1-3000).

**Assembly** (all 16 `check_streamk` configurations x bias on / off x five
tiles, 160 builds): every build changes, as with entry 10's branch-free
attempt (`x * per - max(x - tall, 0)`).
- dp builds: 23-62 fewer instructions and 2-8 fewer SGPRs on every tile.
- Most Stream-K builds: 1-50 fewer instructions; spread at 128x256x64 and
  256x128x64 up to 168 fewer.
- 256x256x64 capped / rs / l2 / owner: 120-134 more instructions, VGPRs +4 or
  level, no new spills.
- Spills rise only on 256x256x64 ragged_m (bias 4 to 10, no bias 0 to 3).
  64x64x64 ragged_m with bias goes from 112 to 284 VGPRs.
- SGPR spills (to VGPR lanes) move on 92 Stream-K builds, -5 to +12 (72 up),
  most on spread (31 to 43); none on dp.

**Checks**: `run_checks.sh` and `--extra` pass on all five tiles (125 cases,
graph replay, both negative controls).

**Timing** (`results/xcd_remap_ab.md`, timing.py's method, 14 points, two runs):
new / branch is 0.996-1.004 everywhere, against 0.2-1.7% round spreads,
including the points whose assembly grew: 256x256x64 capped, owner and rs at
4352x4096x8192 (0.997-1.002), ragged_m with bias at 256x256x64
(1.002 / 1.004), and 64x64x64 (0.998). Within noise, so the branch-free form
is kept.

## 13. The v9full tile order (2026-10-28)

**Decision**: adopted for simplicity, despite entry 11's gate. The Stream-K tail
becomes separate from the persistent mapping, which a configurable per-shape XCD
mapping can replace later.

**Code**:
- `_get_pids` in `persistent_tile` and in the kernel's prologue remaps over
  `total_full_tiles` rather than `total_tiles`, and the next-tile clamp is
  `total_full_tiles - 1`.
- `streamk_tile_id(t, total_full_tiles)` is `total_full_tiles + t`, replacing
  the enumeration of each XCD block's leftover tiles. `persistent_tile` and
  `streamk_first_segment` no longer take `total_tiles`. `xcd_blocks`, which
  only `streamk_tile_id` used, is removed.
- dp (S = 0) and launches without a persistent phase compile as before.

**Checks**:
- Assembly: identical to the `/tmp/v30order/v30_v9full` variant (built from
  entry 12's kernel) on the ten builds of entry 10 (dp aligned and ragged,
  one-tile rs / l2 / ragged l2 / owner / ragged owner with bias, two-tile,
  ragged spread, 64x64x64 rs). The timings of entries 10-11 therefore apply to
  this kernel. The dp builds are identical to entry 12's.
- `run_checks.sh` and `--extra` pass on all five tiles (125 cases, graph replay,
  both negative controls).
- Coverage (entry 11): persistent tiles plus `total_full_tiles + t` cover every
  tile exactly once over 24288 cases.

**Costs against v9's order** (entries 10-11, cold inputs unless noted):
- 256x256x64, the tile the auto rule picks for these shapes:
  - one-tile S=32 on 16x18 tiles, 0.6-1.1%: v9's blocks happened to line up
    with the swizzle groups;
  - spread at K = 16384, 0.8-1.5%: LLC re-reads, 1.73x the compulsory A + B;
  - rows that are not 128-byte aligned with a row-shaped tail, 3.2-3.7%
    (padding the stride removes this and the 27% stride cost).
- Smaller tiles, mostly at shapes where the auto rule would pick 256x256x64:
  - spread at K = 16384, up to 3.5% (128x256x64), LLC re-reads, half of it left
    when the inputs are warm in the LLC;
  - two-tile with two trips, 2.6% (128x256x64) and 6.5% (64x64x64, a 52-tile
    problem at 256x256x64), both gone with warm inputs;
  - one-tile capped / l2 at S=16, 1-2%.

**Gains**: two-tile with one trip (up to 4.7%, 9% on ragged 128x256x64),
one-tile S=16 at 256x256x64 (up to 2%), 129x6 tiles (1.6-3.5%), 256x256x64
two-tile with two trips (2%).

**Follow-up**: a per-shape XCD mapping, tuned with cold inputs, checked against
the points above. It should also control the tail's shape, and use the LRU
model of entry 11 to prune candidates whose waves spill the LLC.
