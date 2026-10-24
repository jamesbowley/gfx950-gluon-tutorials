# v29_streamk_compact work log

v29 is v28 (`v28_streamk_owner`) with only the code paths that runs take: data-parallel, one-tile (owner and
reduce-scatter endings), two-tile and spread, ragged M / N / K, small tiles, the capped rule, `tile="auto"` and the l2
exchange. Everything else v28 carried from earlier experiments is gone; [../DROPPED.md](../DROPPED.md) records what
was tried and dropped, and why, so that it is not tried again. Everything here ran on HIP 1 (amd-smi GPU 0, PCI
0000:05:00.0; [env.sh](env.sh)), on v28's stack, on 2026-10-20.

## 1. The build (2026-10-20)

- **Files:**
  - [matmul_kernel.py](matmul_kernel.py): the kernel `v29_streamk_compact` and its host `matmul`.
  - [asm_diff.py](asm_diff.py): v29's compiled code against v28's.
  - [check_streamk.py](check_streamk.py) and [run_checks.sh](run_checks.sh): the stress check.
  - [timing.py](timing.py): v29 against v28.
- **Size:** 2,138 lines against v28's 3,709 (58 functions).
  - The plan's estimate was 1,700-1,900. The rest is the main loop, the persistent epilogue, the small-tile body
    and the host, which are copied unchanged in substance.
  - The largest functions are the kernel (255 lines), `matmul` (181), `small_segment` (170) and `persistent_tile`
    (168).
- **Knobs kept:**
  - `STREAMK_POLICY` = `capped` (default) | `dp` | `one_tile` | `two_tile` | `spread`.
  - `STREAMK_FIXUP` = `auto` | `owner` | `rs`. With an explicit policy, `auto` is v28's rule: rs if every leftover
    tile is split at least 8 ways, else owner (`default_fixup`).
  - `STREAMK_EXCHANGE` = `wt` | `l2`, plus `STREAMK_NUM_PROGRAMS`, `STREAMK_TILE` and `STREAMK_FORCE_EDGE`.
- **Removed:**

  | what | where it lived in v28 |
  |---|---|
  | bf16 and bf16-pair partials (`STREAMK_PARTIALS`) | `pair_layout`, `to_pairs`, `from_pairs`, `pair_offsets`, the pair branches of the owner and rs endings, the small tiles' bf16 store, the bf16 view of `P` |
  | the atomic ending (`STREAMK_FIXUP=atomic_iz`) | `atomic_quadrant`, `atomic_layout`, `zero_layout`, `zero_rows`, `zero_tile_rows`, `small_atomic_layout`, `zero_small_tile`, `ATOMIC_C`, `ATOMIC_EVEN`, the zero counters (`tile_sync` is now 3 words per program, not 5) |
  | chunk-major placement (`STREAMK_PLACEMENT`) | `streamk_chunk_major_of`, `streamk_slot`, `CHUNK_MAJOR`, the role rotation: role is the chunk, slot is the spid |
  | l2 without the XCD-local split (`STREAMK_XCD_LOCAL=0`, `STREAMK_UNSAFE_PLACEMENT`) | `tile_groups`, `first_straddling_tile`: the XCD-local split cannot straddle, and the host asserts `ceil(S / 8) n <= 32` |
  | `STREAMK_BALANCE_XCDS=0` | `streamk_compact_pid`'s unbalanced branch |
  | the `auto` and `split_rule` policies | `streamk_policy`, `split_rule_policy`, `SPLIT_RULE_TABLE` |
  | all-reader reduce-scatter (`STREAMK_RS_READERS=all`) and v20's C store | `RS_BAND`, `reduce_scatter_slice`'s row / column store |
  | `STREAMK_PERSISTENT_PROGRAMS` | the VGPR half of `check_residency`, `build_vgprs` (now in [../TODO.md](../TODO.md) for stream-K) |
  | `GEMM_TILE_LIMIT`, `STREAMK_RECORD_PLACEMENT`, `STREAMK_P_PAD`, `PERSISTENT_TILE_ORDER=split`, `V9_TILE_GM`, `STREAMK_COMPACT_GRID=0`, the `STREAMK_OWNER_OPT` guard | `TILE_LIMIT`, `record_placement`, `getreg`, `P_SLOT_PAD`, `TILE_ORDER_V9`, `persistent_tile_id`; `MASK_TAIL_PREFETCH` (always off) |
  | v27's per-quadrant ending | `fixup_quadrant` and its 8 calls: the owner ending is always `owner_fixup_tile`'s (now `owner_ending`), reduce-scatter's partial and whole-tile stores are `store_partial` / `owner_store` (`rs_ending`) |
  | the version-history docstrings and "vNN entry M" comments | replaced by a module docstring and one docstring per function |

- **Structure:**
  - **File order:** program and tile mapping, then the quadrant main loop, ragged edges, synchronisation, the
    quadrant partials and endings, `quadrant_segment` and `persistent_tile`, the small-tile body, the kernel, and
    the host.
  - **Ending dispatch:** each segment body ends in one branch on the ending.
  - **Tuples:** related values travel as tuples:
    - `accs`: the four accumulators;
    - `smem`: the four half-tile buffers, or the small tile's strips and views;
    - `aop` / `bop`: an operand's six offsets;
    - `kofs`: the kernel-wide offsets;
    - `strides`;
    - `c_bases`: the four quadrants' C bases.

  The owner and reduce-scatter endings loop over quadrants with `static_range` and index those tuples. The tuples
  are flattened by the frontend, so they cost nothing in the compiled code (section 2).

## 2. Compiled code against v28 ([asm_diff.py](asm_diff.py), [asm_diff.md](results/asm_diff.md))

- **The builds:** 25 per tile on 9 tiles, each compiled in a fresh process with an empty cache, for both versions
  (450 compiles, about 1 minute on 64 jobs).
  - **Tiles:** 256x256x64, 256x128x64, 128x256x64, 128x128x64, 128x128x128, 16x256x128, 32x128x128, 128x32x128 and
    16x16x256.
  - **Configurations:**
    - dp and cap;
    - owner on 2, all and 16 programs per tile;
    - rs on 8 and all;
    - l2;
    - two-tile and spread;
    - bias on 6 of them.
  - **Shapes:** without a full wave, with one, and both with M, N and K ragged.
- **The first form was not identical on most stream-K builds,** from the first few instructions on.
  - **Cause:** the order of operations, not their content. The kernel computed `total_full_tiles` at its top and
    built the kernel-wide offsets in another order. The TTGIR had the same operations, but LLVM allocated
    registers differently.
  - **Fix:** with v28's order restored, those builds are identical.
- **Storing all four partial quadrants first was about 0.9% slower on 256x256 two-tile.**
  - **What changed:** v28 kept v27's per-quadrant order (store a quadrant, read its peers, store its C) where the
    owner reads one quadrant per batch in a run-time peer loop. That is 256x256 end to end, and above 8 programs
    per tile. v29's first form stored all four quadrants first, as `owner_fixup_tile` does elsewhere.
  - **Effect:** the compiled code changed, with the same spills or fewer (spread with bias, ragged: 79 to 50).
    Timed in three sessions, two-tile came out 1.014 / 1.017, then 1.008 / 1.009 and 1.009 / 1.009 (without /
    with bias), against 0.999-1.000 for the cap control.
  - **Fix:** `owner_ending` now stores each quadrant just before its batch in that case (`STORE_EACH`), and those
    builds are v28's code again. The first form's comparison is kept in
    [asm_diff_first_form.md](results/asm_diff_first_form.md).
- **Final result (225 builds):**
  - 217 identical;
  - 3 sent to `v9_tile` by both;
  - 5 different, all accounted for.
- **The 5 differences:**
  - **3 are compiler nondeterminism.** A fresh build of either version gives either form:
    - 128x256x64 two-tile with and without bias, and 256x256x64 spread: a `v_bitop3_b32` operand swap with the truth
      table to match;
    - 128x128x64 `rw0 ow2`: identical in 2 of 4 repeat builds, different in the other 2, in both directions.
    - v28 WORKLOG §2 describes the same forms.
  - **2 differ only in where one merged `make_range` lands:** 256x256x64 two-tile with bias and 128x128x64 rs on
    all programs with bias.
    - **Cause:** v28's `fixup_quadrant` created the bias columns' range at its top, before the partial map's
      equal range. v29 creates it in `owner_store`, so CSE keeps the other one first.
    - **Effect:** same instruction multiset (2,432 instructions at 128x128x64, the same count of every opcode) and
      the same VGPRs; both timed level (section 4).

## 3. Stress ([check_streamk.py](check_streamk.py), [run_checks.sh](run_checks.sh), [summary.txt](results/checks/summary.txt))

- **The check:** v28's `check_streamk.py`, cut down to v29's options.
  - **Configurations:** one-tile with rs and owner, wt and l2 (rs only), two-tile and spread, with and without
    bias. The one-tile split runs at 4S, 4S + S/2 and all programs.
  - **Each configuration:** 10 launches over 3 rotating inputs. C is filled with NaN inside a sentinel window,
    and every owner flag and barrier count must be back to zero afterwards.
- **Coverage:** the 9 tiles of section 2, plus the no-full-tile sets (compact launch) on 256x256x64, 128x128x128
  and 16x256x128.
- **All 1,304 configurations PASS.** After the `STORE_EACH` fix, 256x256x64 and its no-full set (the only tile it
  can change: `QPB = 1`) were re-run: 198 configurations, all PASS.
- **Negative control (`--stale-count`, 256x128x64):** 29 of 32 configurations hang, so it FAILs as required.

## 4. Timing against v28 ([timing.py](timing.py), [timing.md](results/timing.md))

**Method:** `hbl_current.py`'s. Per point and version, a CUDA graph of 100 launches over 2 GB of rotating inputs;
9 rounds of 5 replays in shuffled order, in one process; the median per launch.

| point | tile | shape | v28 us | v29 us | v29 / v28 |
|---|---|---|---|---|---|
| owner, all programs (anchor) | 128x128x128 | 1024x1024x7168 | 27.96 | 27.85 | 0.996 |
| owner, 3 per tile | 128x128x128 | 1024x1024x7168 | 29.95 | 29.84 | 0.996 |
| owner, 2 per tile | 128x128x128 | 1024x1024x7168 | 37.75 | 37.72 | 0.999 |
| rs, all programs | 128x128x128 | 1024x1024x7168 | 27.29 | 27.23 | 0.998 |
| rs, all programs, l2 | 128x128x128 | 1024x1024x7168 | 28.08 | 27.89 | 0.993 |
| owner, 16 per tile | 256x256x64 | 4096x4352x8192 | 244.33 | 246.21 | 1.008 |
| two-tile | 256x256x64 | 4096x4608x8192 | 218.47 | 219.13 | 1.003 |
| two-tile, bias | 256x256x64 | 4096x4608x8192 | 223.83 | 223.82 | 1.000 |
| spread | 256x256x64 | 4096x4608x8192 | 234.19 | 235.54 | 1.006 |
| spread, bias, ragged | 256x256x64 | 4072x4560x8176 | 292.04 | 291.84 | 0.999 |
| rs, all programs, bias | 128x128x64 | 2048x2304x8192 | 86.93 | 86.91 | 1.000 |
| cap (`v9_tile` in both: control) | 256x256x64 | 4096x4096x4096 | 94.70 | 94.80 | 1.001 |

- **Everything is level:** 0.993-1.008, against a 0.6-3.1% spread of each point's own rounds.
- **The changed builds:** the two that compile differently (two-tile with bias, rs with bias) read 1.000 and 1.000
  here.
- **Owner on 16 per tile:** it is v28's code again; two more sessions read it at 1.002 and 1.001, so its 1.008 here
  is noise.
- **Two harness notes:**
  - **Keep each graph's inputs alive.** The first `timing.py` freed each point's rotating copies when it built the
    next point's graph, and the earlier graphs then read freed memory: a GPU memory fault, reported in the v29
    kernel. It now keeps every point's copies; `hbl_current.py` already did.
  - **The l2 exchange is safe under CUDA graphs.** Each captured launch has its own epoch, and a flag only has to
    differ from the previous launch's.
