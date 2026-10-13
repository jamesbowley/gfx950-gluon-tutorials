# v24_streamk_combined work log

One entry per change: what changed and what it did to performance and spills. Results come
from Triton v2.2 + deepbind shim on `PYTHONPATH` and
`LLVM_PASS_PLUGIN_PATH=plugins/llir_scheduler/libLlirSched.so`, **without** the amdgcnas hook
(as v22, v23 and sweep C), run from `a16w16/`, on GPU 0. Raw tables are in `results/` and
[experiments/streamk_split_count/results](../../../../../experiments/streamk_split_count/results/).

## 1. Every ending, placement and partial type in one kernel (2026-10-08)

v23 with three options from earlier versions, so that one kernel runs every ending, placement and
partial type and a sweep can change one thing at a time. With the new options off the kernel is
v23's (below). All three are for the tile-aligned one-tile split only.

**The changes.**
1. **Atomic ending** (`STREAMK_FIXUP=atomic_iz`, from v22). Every program of a stream-K tile adds its
   partial into C with `buffer_atomic_pk_add_bf16`, after zeroing 1/n of the tile's rows itself and
   waiting on the tile's count of zeroed shares. Only the in-GEMM zeroing is ported: v22's separate
   zeroing launch was a median 13% longer at the same n in sweep C. The host asserts 16-bit C and
   `STREAMK_PARTIALS=fp32` (the ending has no partials).
2. **Partials through one XCD's L2** (`STREAMK_EXCHANGE=l2`, from v21, reduce-scatter only). Default-policy
   partial stores, an epoch flag per program polled with `.cg` (`sc0 nt`) loads instead of the barrier's
   atomics, `.cg` slice loads, and v21's XCD-local split (`STREAMK_XCD_LOCAL`, on by default with `l2`),
   which puts every tile's programs on one XCD and caps n so that ceil(S / 8) tiles fit in an XCD's 32
   programs. The host replays the placement and asserts that no tile spans two XCDs
   (`STREAMK_UNSAFE_PLACEMENT=1` skips that, for the negative control). v21's `STREAMK_PARTIALS=wt|l2` is
   renamed `STREAMK_EXCHANGE` because v23 uses `STREAMK_PARTIALS` for the partial type; `mem` is dropped.
   v21's L2 path now runs under v23's slice (band store, power-of-two readers) and with bf16 partials.
3. **Chunk-major placement** (`STREAMK_PLACEMENT=chunk`, from v18). Program w runs chunk j = w // S of tile
   t = w % S (the uneven tiles' extra chunk after all the others), so an XCD holds the same chunk of
   neighbouring tiles. The tile's role 0 (owner, first reduce-scatter slice, bias) rotates to chunk
   t % n: chunk-major puts every tile's chunk 0 on the same XCDs, and a fixed owner cost v18 up to 8%.
   Under tile-major placement (all other versions) a tile's programs are consecutive ids on one or two
   XCDs, so role 0 is spread over the XCDs without a rotation.

**One mechanism for the placements.** Every program now carries a role (0 .. n - 1) and a workspace
slot, `streamk_slot(t, role)`: the id the role would have under tile-major placement. The endings use
the slot and the role where v23 used the stream-K id and the chunk index (partial slots, flags and
their reset, the barrier index, the reader index, the zero counts), so they read the same slots and
barriers whichever program runs a role. Tile-major placement passes slot = id and role = chunk, which is
v23's code.

**A bug found on the way, in the atomic ending's workspace.** v22's zero counts used the reduce-scatter
barrier's words, `tile_sync[2b]` and `tile_sync[2b + 1]`. The second is the barrier's generation, which
only ever goes up. Once a reduce-scatter launch had run in the same process, the atomic ending's count of
programs past their adds started from that generation, so the reset could fire while the tile's other
programs were still waiting for the zeroed count, and they waited forever (the first asm check hung on
its first atomic chunk-major case, after the reduce-scatter cases). v22 never hit it: its rocprof and
check workers never ran both endings in one process. v24 gives the zero counts their own words,
`tile_sync[3 * 256 + 2b]` and the one after (`tile_sync` grows to 5 x 256 words: the barriers' pairs, the
L2 epoch flags, the zero counts). [v22](../v22_streamk_atomic_c/matmul_kernel.py) still shares them; a
process that runs v22's reduce-scatter and then its `atomic_iz` can hang.

**Checks** (`results/`):
- `asm_check.py` ([asm_check.md](results/asm_check.md)), 4096x8192xK with `GEMM_TILE_LIMIT = 256 + S`:
  - **against v23**, ten v23 configurations (owner and reduce-scatter, even and uneven tiles, fp32, bf16
    and bf16 pairs, both reader modes, two-tile, spread, dp, the defaults): the TTGIR is identical apart
    from the signature's extra `epoch` argument, with the same VGPRs and spills. Nine compile to identical
    instructions; in the bf16-pairs reduce-scatter build LLVM orders one integer address expression in the
    slice set-up differently (74 of 4782 lines, 6 fewer instructions), a side effect of the extra argument;
  - **v24's options**, 24 builds (owner, reduce-scatter and atomic_iz, tile- and chunk-major, fp32 and
    bf16 pairs, even and uneven tiles, and `l2`): no spills, 444-484 VGPRs (chunk-major owner 484, the
    rest within v23's 444-480). atomic_iz: 128 `buffer_atomic_pk_add_bf16`, no partial loads, and its only
    write-through stores are the zeroing's `dwordx2`. `l2`: partial stores with no cache bits and every
    slice and flag load `sc0 nt`.
- `check_blocks.py` ([check_blocks.md](results/check_blocks.md), pure Python): v23's address-map checks,
  then for every S, program count S..256 and K = 512, 2048, 8192 under both placements, the kernel's decode
  replayed for every stream-K id: unique (tile, chunk), every tile's chunks and roles 0..n - 1 exactly
  once, its slots exactly `streamk_slot(t, 0) .. + n - 1`, tile-major slot = id and role = chunk, chunk-major
  role 0 = chunk t % n; and for every S and n the XCD-local split keeps each tile on one XCD.
- `check_streamk.py` ([check_streamk.md](results/check_streamk.md)): 376 configurations, 30 launches each
  over 3 rotating inputs, on 4352x4096x8192 (S = 16), 4352x4352x8192 (S = 33, uneven tiles), 3328x5120x8192
  (S = 4) and 4096x6144x8192 (S = 128), with and without bias: every valid combination of ending, fp32 or
  bf16-pairs partials, reader mode, placement and exchange, at n = 4, at an uneven split and at the
  default, plus two-tile. No wrong results; every flag, barrier count and zero count re-armed. bf16
  partials and atomics are held to an fp32 reference by each tile's mean error (1 + sqrt(n) roundings).
  Negative controls, both detected:
  - `--stale-count` ([check_streamk_stale.md](results/check_streamk_stale.md)): tile 0's barrier count
    preset to n - 1 hangs all 12 reduce-scatter configurations, tile- and chunk-major;
  - `--unsafe-placement` ([check_streamk_unsafe.md](results/check_streamk_unsafe.md)): `l2` with the
    balanced placement on 3328x5120x8192 (16 programs per tile, 8 per XCD) gives wrong results on all 30
    launches of both reader modes.
- `check_v9_fallback.py` ([check_v9_fallback.md](results/check_v9_fallback.md)): 0 failures.
- `check_tile_limit.py --sweep f --versions 24`
  ([check_tile_limit_f.md](../../../../../experiments/streamk_split_count/results/check_tile_limit_f.md)): all
  5,701 of sweep F's v24 cases write exactly the first T tiles, match torch (bf16 partials and atomics by
  mean error) and re-arm every flag, barrier count and zero count.

## 2. Sweep F and the held-out AITER test (2026-10-09)

**Sweep F** ([experiments/streamk_split_count](../../../../../experiments/streamk_split_count/README.md), section 5;
[sweep_f.md](../../../../../experiments/streamk_split_count/results/sweep_f.md)): sweep C's matrix
(4096x8192xK, `GEMM_TILE_LIMIT = 256 + S`) with every v24 option, S = 1-128 at K = 512-16384 and every n, plus
v13, two-tile and v20's reduce-scatter; 6,434 cases, rocprof, GPU 0 (HIP numbering) when idle. Tail ratios,
each configuration at its best n, median (10th-90th percentile) over the 90 cells with S <= 128:

| comparison | ratio | first shorter in |
|---|---|---|
| bf16-pair / fp32 partials, reduce-scatter | 0.856 (0.758-0.987) | 85 of 90 |
| bf16-pair / fp32 partials, owner | 0.926 (0.886-1.020) | 79 of 90 |
| atomic_iz / bf16-pair reduce-scatter (both round more than once) | 1.038 (0.946-1.193) | 25 of 90 |
| atomic_iz / fp32 reduce-scatter | 0.882 (0.772-1.134) | 78 of 90 |
| chunk-major / tile-major, reduce-scatter | 0.992 (0.934-1.040) | 52 of 90 |
| chunk-major / tile-major, owner | 1.009 (0.971-1.065) | 34 of 90 |
| l2 with the XCD-local split / write-through, reduce-scatter | 0.989 (0.861-1.249) | 50 of 90 |
| v24 / v20 reduce-scatter | 0.972 (0.907-1.025) | 60 of 90 |
| reduce-scatter / owner (fp32) | 0.877 (0.376-1.047) | 77 of 90 |
| two-tile / one-tile reduce-scatter | 1.491 (1.137-1.955) | 0 of 90 |

- **bf16 partials are the largest lever**, 15-20% off reduce-scatter's tail at K >= 2048.
- **The atomic ending is not worth its roundings**: like for like it is shorter than bf16 reduce-scatter in only
  25 of 90 cells (median 3.8% longer), as v23's E1 found on seven. Its wins are mostly at large S and K, where
  n is 2-6.
- **Chunk-major placement is neutral** for every ending (v18's finding holds with the split capped).
- **l2 depends on K and S**: a median 5-8% shorter at K <= 1024 (up to 29%), 20-47% longer at S = 2-8 for K >= 4096 (where the
  XCD-local split packs each tile on one XCD instead of spreading it over up to eight), mixed at K = 16384.
  Sweep F cannot separate the exchange from the placement, which `l2` always brings.
- **v24's slice is 3% faster than v20's** over the grid (5-8% at K = 2048-8192).
- Run-to-run spread ([sweep_f_round2.jsonl](../../../../../experiments/streamk_split_count/results/sweep_f_round2.jsonl):
  the 501 most competitive choices timed again): kernel times repeat to 0.3% (median) and 1.05% (90th
  percentile), so differences under about 1% are ties.

**Rules** ([rules_f.md](../../../../../experiments/streamk_split_count/results/rules_f.md)). Kernel-time regret
against the fastest choice among the rule's own configurations, over sweep F's 90 cells: v24's `capped` rule
1.000 / 1.044 / 1.111 (median / 90th percentile / worst). A six-constant rule fitted to the cells (data-parallel
below 12 idle k-steps, reduce-scatter from 6 programs per tile at 4 k-steps per program and at most 128
programs, otherwise owner with c = 3) reaches 1.001 / 1.039 / 1.086 in sample but 1.052-1.058 at the 90th
percentile cross-validated (one S or one K left out): no better than `capped`. Frozen in
[rules_frozen.json](../../../../../experiments/streamk_split_count/results/rules_frozen.json) before the test.

**Held-out test: AITER's shapes** ([experiments/streamk_aiter_test](../../../../../experiments/streamk_aiter_test/README.md),
[test.md](../../../../../experiments/streamk_aiter_test/results/test.md)). The 440 AITER bf16 shapes with a
partial last wave, every single-rounding choice timed (data-parallel as v9 at one wave or less, else v13; v24
reduce-scatter and owner at every n; v24 as deployed; the frozen rule), contiguous B and C, rocprof; correctness
of all 4,738 cases checked first. Kernel time / the fastest of these for the shape:

| shapes | count | data-parallel | v24 capped (deployed) | fitted rule (frozen) |
|---|---|---|---|---|
| less than one wave | 314 | 1.464 / 3.124 / 4.997 | 1.029 / 1.161 / 1.264 | 1.044 / 1.154 / 1.435 |
| one full wave or more | 126 | 1.041 / 1.214 / 1.417 | 1.003 / 1.012 / 1.040 | 1.003 / 1.025 / 1.076 |

- **With a full wave, `capped` is within 1.2% of per-shape tuning at the 90th percentile**, 4% at worst. The
  fitted rule is level at the median and slightly worse in the tail, as cross-validation predicted.
- **With less than one wave, both rules lose up to 26% (capped) and 43% (fitted).** Every shape there is outside
  sweep F, whose tails all follow a full wave. Most of the loss is one decision: `capped` switches to the owner
  ending from S = 33 (fewer than 8 programs per tile possible) and the fitted rule from S = 43 (fewer than 6),
  while up to S = 64 the fastest choice is reduce-scatter at n = 4-6 (shapes with S < 64: `capped` median 1.091, 90th percentile 1.188; S >= 128: 1.001
  and 1.015). With no full wave to overlap, reduce-scatter's flat ending pays from fewer programs per tile than
  after one. The fix needs a sweep without a full wave (`GEMM_TILE_LIMIT = S`), fitted there and tested on
  AITER again; tuning it to these shapes would spend the test set.
