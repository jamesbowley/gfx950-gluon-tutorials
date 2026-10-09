# v23_streamk_rs_band work log

One entry per change: what changed and what it did to performance and spills. Results come
from Triton v2.2 + deepbind shim on `PYTHONPATH` and
`LLVM_PASS_PLUGIN_PATH=plugins/llir_scheduler/libLlirSched.so`, **without** the amdgcnas hook
(as v22's log and v20 entry 14), run from `a16w16/`. Timing on GPU 1 with no other user;
checks on GPUs 0-7. Raw tables are in `results/` and
[experiments/streamk_split_count/results](../../../../../experiments/streamk_split_count/results/).

## 1. Band C store, power-of-two readers, bf16 partials for comparison (2026-10-08)

v20 at WORKLOG entry 14 (write-through partials, relaxed flags and tile barrier), with what
[experiments/rs_slice_layout](../../../../../experiments/rs_slice_layout/README.md) found about the
reduce-scatter slice. v20 is unchanged.

**The changes.**
1. **Band C store** (`reduce_scatter_slice`). After summing a group of `NB` aligned blocks in the
   slice layout, the group is reshaped to its 2-D band (`to_band`) and converted to a row store
   layout before the C store: `gStoreLayoutC` for 128 columns (NB >= 4), 8 lanes x 16 B per row for
   NB = 2, 8 lanes x 8 B for NB = 1. Each lane stores 16 contiguous bytes of a C row (8 at NB = 1)
   instead of v20's 8 bytes with 4 neighbouring lanes on 4 rows. The bias is still added in fp32
   in the slice layout, before the single rounding.
2. **Power-of-two readers** (`STREAMK_RS_READERS=pow2`, the default; `all` is v20's). Of a tile's
   n programs only the first R = 2^floor(log2 n) read: reader j sums blocks
   `[64 j / R, 64 (j + 1) / R)` of all n partials. The other n - R publish, arrive on the barrier
   and finish (`tile_barrier(..., wait)`). The host computes R for the tiles with N_HI and N_LO
   programs (`rs_split`) and passes `RS_R_HI` / `RS_R_LO`; the kernel asserts they match its own
   program counts. With power-of-two readers every slice is whole groups: `NB = 64 / R_max` (at
   most 16), and `RS_BAND` is set. With `all`, the band store is used only when every n is a power
   of two; otherwise v20's store and v20's group size.
3. **bf16 partials** (`STREAMK_PARTIALS=fp32 | bf16 | bf16_pairs`, default **fp32**), for both
   endings. **Opt-in, for comparison only**: the earlier comparison between reduce-scatter and the
   atomic ending had fp32 partials on one side and bf16 adds on the other. These options give the
   reduce-scatter and owner endings bf16 traffic too.
   - `bf16`: v20's lane-contiguous map with bf16 values, 8 B per lane (`dwordx2`), 512 contiguous
     bytes per wave instruction.
   - `bf16_pairs`: a lane's 8 registers of two blocks 32 columns apart as 16 contiguous bytes
     (`to_pairs` / `pair_offsets` / `from_pairs`, from `rs_slice_layout/split_kernel.py`), `dwordx4`.
     The owner loads in the pair layout and reshapes to the MFMA layout for free; reduce-scatter
     reads its band in the pair layout when NB >= 4, and in the slice layout with the pair map's
     per-block offsets otherwise (8 B per lane).
   - P is the same buffer, through a bf16 view; each slot uses half its bytes. Sums stay fp32.

**Checks** (`results/`):
- `check_blocks.py` (pure Python): v20's decode and coverage; the pair map's per-block form
  agrees with `pair_offsets` for all 65536 elements and gives 4 aligned consecutive elements per
  lane; R readers cover a tile for n = 2-128; `rs_split` gives whole aligned groups for all
  460,544 combinations of S, program count, K and reader mode.
- `asm_check.py` ([asm_check.md](results/asm_check.md)), one-tile on 4096x8192x16384, NB = 16 to 1:
  - the slice's C store is `buffer_store_dwordx4` (`dwordx2` at NB = 1); v20's is `dwordx2`;
  - one conversion per group: LDS bytes written = bytes read = the group's bf16 values, at most 2
    rounds; 8 `ds_write_b128` + 8 `ds_read_b128` and 4 `s_barrier` at NB = 16;
  - slice loads `dwordx4` (fp32), `dwordx2` (bf16), `dwordx4` (bf16 pairs at NB >= 4); partial
    stores `dwordx4` (fp32, pairs) or `dwordx2` (bf16), all write-through;
  - 444-480 VGPRs, no spills (v20 448-480).
- `check_streamk.py` (v20's, over every `STREAMK_PARTIALS` value and, for one-tile reduce-scatter,
  both reader modes; 30 rotating launches each):
  - fp32: [144 configurations](results/check_streamk_fp32.md), all correct, every flag and count
    re-armed;
  - bf16 and bf16 pairs: [144](results/check_streamk_bf16.md) + [144](results/check_streamk_bf16_pairs.md),
    all correct with every flag and count re-armed. Held to an fp32 reference by each tile's mean
    error (at most 1 + sqrt(n) times one rounding's, the test check_tile_limit.py uses for the
    atomic ending). Under bench.py's `allclose` 13 and 12 configurations fail
    ([bf16](results/check_streamk_bf16_allclose.md), [pairs](results/check_streamk_bf16_pairs_allclose.md)):
    all are 2-way splits (4096x6144x8192, 8192x7936x8192, 3840x4096x8192), in both reader modes,
    on one input in three, deterministically. A half-K partial rounded to bf16 puts a few large
    elements past atol 0.1 / rtol 0.01; largest error 0.215 against 0.125 for fp32 partials.
  - `--stale-count` negative control ([results](results/check_streamk_stale.md)): all 12
    configurations hang, as they must (n = 16, 7/8 and 64; fp32 and pairs; both reader modes).
- `check_v9_fallback.py` ([results](results/check_v9_fallback.md)): 0 failures.
- `check_tile_limit.py --sweep e1` ([results](results/check_tile_limit_e1.md)): all 163 E1 cases,
  NaN-filled C, exact tiles written, flags re-armed; bf16 partials by mean error.
- Accuracy ([atomic_accuracy_v23.md](../../../../../experiments/streamk_split_count/results/atomic_accuracy_v23.md),
  stream-K tiles of 4096x8192xK against an fp32 reference, S = 1-128, K = 2048-16384), mean error
  over that of one bf16 rounding:

  | n | 2 | 4 | 8 | 16 | 32 | 64 |
  |---|---|---|---|---|---|---|
  | fp32 partials (v20, v23) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
  | bf16 partials, reduce-scatter | 1.53 | 1.56 | 1.58 | 1.60 | 1.61 | 1.61 |
  | bf16 partials, owner | 1.27 | 1.44 | 1.53 | 1.57 | 1.60 | 1.60 |
  | atomic (v22) | 1.53 | 2.03 | 2.61 | 3.40 | 4.54 | 6.20 |

  - bf16 partials stay at about 1.6 times one rounding whatever n: each partial is rounded once,
    and their errors add like the partials themselves. The owner's own partial stays fp32.
  - The atomic ending grows as 0.5 + 0.75 sqrt(n): its running sum is rounded at every add.
  - Outside atol 0.1 / rtol 0.01 in this grid: bf16 partials none (max error 0.25); atomic 0.05%
    at n = 8, 0.33% at 16, 1.8% at 32, 6.2% at 64.
  - bf16 pairs are bitwise identical to bf16 (the same values in another map).

**E1: kernel tails** (`experiments/streamk_split_count/sweep.py e1`,
[e1_tails.md](../../../../../experiments/streamk_split_count/results/e1_tails.md)). The cells where
the atomic ending won in the GEMM grid ([tail_variants.md](../../../../../experiments/memory_model_litmus/results/tail_variants.md))
but lost in the synthetic exchange, on 4096x8192xK with `GEMM_TILE_LIMIT = 256 + S`, each
configuration at the grid's best n and its neighbours. rocprof, 1000 launches over rotating
inputs, median; tail = kernel - the full wave alone; mean of two interleaved rounds (the second
in reverse order). Each configuration at its best n (n in brackets):

| S | K | v20 rs | v23 rs (fp32) | v23 rs bf16 | v23 rs bf16 pairs | v22 atomic | v22 atomic, zeroed in the GEMM |
|---|---|---|---|---|---|---|---|
| 1 | 16384 | 9.7 (64) | 8.9 (64) | **7.2** (64) | 7.9 (64) | 9.8 (64) | 9.1 (64) |
| 8 | 8192 | 22.6 (16) | 22.1 (16) | 18.3 (16) | **17.8** (16) | 22.7 (12) | 19.0 (12) |
| 16 | 8192 | 30.7 (8) | 29.6 (8) | 26.2 (12) | **25.5** (12) | 29.5 (8) | 25.8 (8) |
| 32 | 4096 | 32.7 (4) | 30.4 (4) | **26.1** (4) | 27.1 (4) | 30.9 (3) | 27.1 (3) |
| 64 | 8192 | 67.5 (3) | 63.1 (3) | **54.6** (4) | 55.2 (4) | 58.0 (4) | 54.9 (4) |
| 12 | 8192 | 28.7 (8) | 27.8 (8) | 24.2 (16) | **23.4** (16) | 28.5 (8) | 25.2 (8) |
| 24 | 8192 | 42.1 (6) | 39.7 (6) | 35.8 (8) | 34.8 (8) | 38.7 (6) | **34.7** (6) |

- **The band store**, at power-of-two n (v20 against v23, the same readers): 0.5-3.0 us shorter
  tails in all 11 cells, most at n = 4 (32.7 to 30.4, 68.5 to 65.7, 44.3 to 41.3).
- **Power-of-two readers**, at uneven n (`v23 rs all` against `v23 rs`): 1.6-4.4 us shorter in all 8
  cells (S = 64, n = 3: 67.5 to 63.1). `v23 rs all`, which runs v20's slice there, is level with
  v20 (within -1.0 to +1.0 us), as it should be.
- **v23 against v20, each at its best n:** 0.5-4.4 us (2-8%) shorter tails on every cell.
- **bf16 partials against fp32** (v23 reduce-scatter, each at its best n): 1.7-8.5 us shorter
  (12-19%). The cheaper exchange moves the best n up (8 to 12 or 16, 6 to 8).
- **Like for like, bf16 reduce-scatter against bf16 atomics:** v23 with bf16 partials is as fast as
  the atomic ending zeroed in the GEMM, or faster, on all seven cells (7.2 against 9.1 at S = 1,
  17.8 against 19.0 at S = 8, level within 0.3 us at S = 16, 64 and 24), and faster than the
  launch-zeroed atomic ending everywhere, while rounding C about 1.6 times instead of
  0.5 + 0.75 sqrt(n) times. **So the atomic ending is ruled out**: its apparent advantage came from
  comparing bf16 adds with fp32 partials. bf16 and bf16 pairs are within about 1 us of each other,
  in either order.
- The owner ending with bf16 pairs is 2-15 us shorter than v20's, still behind reduce-scatter
  at these cells.
- These cells' tails differ from the grid's (measured on GPU 0 before v20 entry 14); compare only
  within this table.

**E2: where the time goes** (`fixup_timing.py --tile-limit`,
[fixup_timing_e2.md](results/fixup_timing_e2.md)). v20's stamping harness, extended: it patches v20,
v22 or v23 (`VERSION:FIXUP[+NAME=VALUE]`), adds a stamp after the slice's last loads (`reads` /
`C store`) and stamps v22's atomic ending (MAC done, zeroing wait, adds complete). Medians over
the readers, 10 launches, each cell at its best n; us:

| S | n | config | publish | barrier wait | reads | C store | ending (MAC done to C stored) |
|---|---|---|---|---|---|---|---|
| 8 | 16 | v20 rs | 5.4 | 1.8 | 4.6 | 0.9 | 12.8 |
| 8 | 16 | v23 rs | 4.7 | 2.0 | 4.8 | 0.4 | 12.1 |
| 8 | 16 | v23 rs bf16 pairs | 2.8 | 1.8 | 2.7 | 0.4 | 7.8 |
| 8 | 16 | v22 atomic_iz | - | 0.4 | - | - | 10.1 (adds 9.6) |
| 32 | 4 | v20 rs | 4.6 | 1.8 | 4.7 | 3.6 | 15.6 |
| 32 | 4 | v23 rs | 5.0 | 1.5 | 4.6 | 0.9 | 12.0 |
| 32 | 4 | v23 rs bf16 | 3.4 | 1.4 | 3.2 | 0.9 | 8.9 |
| 32 | 4 | v22 atomic_iz | - | 0.4 | - | - | 12.0 (adds 11.6) |
| 64 | 4 | v20 rs | 7.6 | 2.5 | 8.5 | 3.6 | 23.0 |
| 64 | 4 | v23 rs | 7.7 | 2.3 | 8.0 | 0.9 | 19.4 |
| 64 | 4 | v23 rs bf16 | 4.4 | 1.8 | 4.9 | 1.0 | 12.4 |
| 64 | 4 | v22 atomic_iz | - | 0.3 | - | - | 22.2 (adds 21.8) |
| 1 | 64 | v23 rs bf16 pairs | 2.3 | 6.2 | 3.4 | 0.4 | 12.5 |
| 1 | 64 | v22 atomic_iz | - | 0.4 | - | - | 7.2 (adds 6.8) |

- **The band store is the C store's whole saving:** 0.4-1.0 us against v20's 0.4-3.6, most where
  each reader stores the most C (n = 4-6: 3.6 to 0.9, 3.1 to 0.8); level at n = 64, where a reader
  stores 2 KiB.
- **bf16 partials cut publish by 30-46% and the reads by 17-44%**: fp32 3.3-7.7 and 3.6-8.0 us,
  bf16 2.3-4.5 and 2.7-4.9.
- **The bf16 reduce-scatter ending (7.8-12.7 us) is shorter than the atomic adds (6.8-21.8 us) on
  six of seven cells.** At S = 1, n = 64 the 64 arrivals on one tile's barrier take 6-8 us, against
  1.4-2.8 elsewhere. There the stamped tail favours the atomic ending (18.9 against 21.6 us) while
  the kernel time favours reduce-scatter (E1: 7.2 against 9.1); the stamps' extra barriers and
  drains weigh most on a 64-way tile, so E1 is the measure.
- **Against the synthetic split sweep** ([rs_slice_layout bf16.md](../../../../../experiments/rs_slice_layout/results/bf16.md)),
  at the same CU count, reads + C store agree to within 0.8 us: S = 16, n = 8 (128 CUs) 5.2 / 3.9 /
  3.6 us (fp32 / bf16 / pairs) against 5.92 / 4.00 / 3.68; S = 64, n = 4 (256 CUs) 8.9 / 5.9 / 5.5
  against 9.72 / 6.04 / 6.04. The GEMM adds no slice cost of its own (VGPR pressure, code size).
  Publish is shorter than the synthetic one (S = 64, n = 4: 7.7 / 4.4 us against 11.3 / 9.7, fp32 /
  bf16), whose writers also generate their values. The barrier wait is longer (1.4-2.8 against
  0.5-1.1 us): the GEMM's programs reach it less together than the synthetic ones, which start
  from a grid-wide barrier.

**E3** (skew and per-CU costs, if atomics still won): not needed. Like for like, the bf16
reduce-scatter ending is as fast or faster, and E2 finds no GEMM-specific slice cost. The remaining
gap between fp32 reduce-scatter and bf16 atomics is the fp32 traffic, not a fair comparison.

**A pitfall in the stamping harness.** The first E2 run gave wrong results for stamped v23
reduce-scatter and v22's atomic ending; the unstamped kernels pass every check. The stamp after
the tile barrier (inline asm `s_barrier; s_memrealtime`) was the cause: inline asm is not marked
convergent, so LLVM sank and duplicated it into the exec-masked blocks the structurizer builds
around the scalar-atomic broadcast (it treats the broadcast VGPR as divergent). Waves then executed
different numbers of barriers and paired the wrong ones. In v20 the same stamp happened to stay
put. That stamp now has no barrier (`_stamp(..., BARRIER=False)`, wave 0's clock); every stamped
configuration is correct. The kernels' own inline barriers (`drain_and_barrier`) sit at uniform
points, before any such branch.

**Defaults:** `STREAMK_RS_READERS=pow2`, `STREAMK_PARTIALS=fp32` (one rounding, as v20).
The host rule (`capped`) is v20's; its crossovers and split counts were fitted to v20's ending and
would move with v23's cheaper slice.
