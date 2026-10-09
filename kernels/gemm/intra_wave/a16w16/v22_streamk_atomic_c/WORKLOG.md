# v22_streamk_atomic_c work log

One entry per change: what changed and what it did to performance and spills. Results come
from Triton v2.2 + deepbind shim on `PYTHONPATH` and
`LLVM_PASS_PLUGIN_PATH=plugins/llir_scheduler/libLlirSched.so`, **without** the amdgcnas hook
(`TRITON_AMDGCNAS_PLUGIN` unset), as AITER builds its kernels: this plugin compiles v13 and v20
to the same instructions as AITER's opt-in llirSched build. GPU 0, run from `a16w16/`. Raw tables
are in `results/` and in [experiments/streamk_split_count](../../../../../experiments/streamk_split_count/README.md).

## 1. An atomic ending: bf16 `pk_add` straight into C (2026-10-07)

v20 with a third ending for the tile-aligned one-tile split, `STREAMK_FIXUP=atomic` (constexpr
`ATOMIC_C`). Every program of a split tile adds its whole partial into C with
`buffer_atomic_pk_add_bf16` (two packed bf16 per lane, relaxed, `gpu` scope), and chunk 0 adds
the bias. There is no partial workspace, no flag, no barrier and no slice: the ending is the MAC
loop, a `convert_layout` to 2 elements per lane with a wave's 64 lanes along one row (256
contiguous bytes per instruction; straight from the MFMA layout the same adds are 3x slower,
[experiments/streamk_atomics](../../../../../experiments/streamk_atomics/README.md)), and the
atomics. The host zeroes the S stream-K tiles of C first in a launch of its own,
`v22_zero_tiles`, which finds the tiles exactly as the GEMM does (`streamk_tile_id`, then the
`GROUP_SIZE_M` swizzle); the kernel boundary orders the zeroing before the adds.

- Only for 16-bit C, and only for the tile-aligned split (asserted). A whole tile (`n = 1`) takes
  the same path: one add into zero.
- The default policy and ending are still v20's; `atomic` is opt-in.
- **Registers.** 484-496 VGPRs, no VGPR or SGPR spills, no scratch; 128
  `buffer_atomic_pk_add_bf16` per thread, without `sc0` / `sc1` (the no-return device-scope form).
  v20's own paths keep their 3-5 SGPR spills (44 at n = 2-4).
- **Correct.** `check_streamk.py --policies one_tile two_tile --fixups rs owner` (the v20 paths of
  this copy): 54 configurations, all correct with flags re-armed. The atomic ending:
  [check_tile_limit_v22.md](../../../../../experiments/streamk_split_count/results/check_tile_limit_v22.md),
  all 458 sweep C cases (S = 1-128, K = 2048-16384, every n). Because C is rounded once per
  program, its stream-K tiles are checked against an fp32 reference by their mean error, at most
  `1 + sqrt(n)` times one bf16 rounding's; every other tile must match as usual. A negative
  control that skips the zeroing (so every partial is added twice) gives 706-713 times, and is
  rejected.

**Accuracy** ([atomic_accuracy.md](../../../../../experiments/streamk_split_count/results/atomic_accuracy.md),
stream-K tiles against an fp32 reference, inputs uniform in [-0.5, 0.5)):
- The mean error is about `0.5 + 0.75 sqrt(n)` times that of one rounding, whatever S and K:
  1.53x at n = 2, 2.0x at 4, 2.6x at 8, 3.4x at 16, 4.5x at 32, 6.2x at 64. v20 rounds once (1.00x).
- Elements outside the repository's check tolerance (atol 0.1, rtol 0.01): none for n <= 8 at
  K <= 4096; 0.04% at n = 8, K = 16384; 0.07% at n = 16 and 0.6% at n = 32, K = 8192; 2.7% at
  n = 64, K = 8192 and 6.1% at K = 16384; 15% at the default n = 128 (S = 1, K = 16384).
- Run to run, 0-89% of the elements differ bitwise (none at n <= 3): the order of the adds is not
  fixed.

**Speed** (sweep C on one 4096x8192xK matrix, tail = kernel time minus the full wave alone; the
atomic call is timed from the zeroing launch's start to the GEMM's end):
- **The fastest tail in 50 of 60 (S, K) cells for S <= 128.** At its best n it is a median 10%
  shorter than reduce-scatter's at S <= 16 (0.88-1.29x), 17% at S = 24-32, and 5% shorter than
  the best single-rounding ending at S = 38-128, where the owner wins among those.
- It loses where the tail is shortest: S = 2-4 at K <= 4096 (to reduce-scatter, by up to 12%) and
  S = 64-128 at K <= 4096 (to the owner, or data-parallel).
- Its best n is reduce-scatter's in almost every cell (16-64 at S <= 4, 8-16 at S = 6-16, 4-6 at
  S = 24-48).
- **The zeroing launch costs 4.4-5.9 us** (no gap before the GEMM): without it the atomic tail at
  S = 8, K = 8192 would be 18.2 instead of 22.7 us.
- L2: the tail reads no more than data-parallel's (S = 8, K = 8192: 0.96 M lines against 0.96;
  reduce-scatter 0.99-1.09 M).

So the atomic ending trades n bf16 roundings and bitwise reproducibility for a 5-20% shorter
tail (entry 2 moves the zeroing into the GEMM, which makes it a median 11-28% shorter than the
best single-rounding ending). Capped at n = 8, its error stays within the check tolerance at
K <= 4096 (0.04% of elements at K = 16384), and it is still the fastest ending from S = 12 to 48
at every K. It is not at S = 6-8 with K >= 8192, where the best split is 16: there reduce-scatter
at n = 16 is 7-21% faster than the atomic ending held to n <= 8.

## 2. Zeroing C inside the GEMM, small K, and packing onto one XCD (2026-10-07)

**`STREAMK_FIXUP=atomic_iz`** (constexpr `ZERO_IN_KERNEL`): no zeroing launch. Before its K loop each of
a tile's n programs zeroes 1/n of the tile's rows with write-through stores (`zero_tile_rows`), drains
in every wave and arrives on the tile's counter (`tile_sync[2b]`, b = the tile's first spid). Before its
adds it spins (relaxed, `gpu` scope) until the count is n, which the K loop has almost always covered.
After its adds it counts itself on `tile_sync[2b + 1]`; the last one resets both, so the counters are
re-armed for the next launch. This is the cross-XCD sequence the litmus experiment found correct:
`.wt` stores, a drain, then a relaxed flag; the atomics then add into the zeros.
- 484-496 VGPRs, no VGPR spills, 0-1 SGPR spills; the counters are zero after every launch.
- Correct: the NaN-filled check of every sweep C case ([check_tile_limit.md](../../../../../experiments/streamk_split_count/results/check_tile_limit.md),
  4,281 cases, 621 of them `atomic_iz` and 194 `atomic_iz` packed, all passing). A missing zero leaves NaN; an add
  before the zeros would lose a partial, which the mean-error test rejects.
- Accuracy: the same adds as entry 1, so the same rounding.

**Speed** (sweep C, K = 2048-16384, S = 1-128):
- **A median 3.2 us shorter than `atomic` at the same n** (1.7-4.1), 3-13% of the tail: the zeroing
  launch cost 4.4-5.9 us, the in-kernel zeroing about 1 us.
- **The fastest tail in 54 of 60 cells.** The best single-rounding tail is a median 1.39x as long at
  S <= 16 (1.16-1.86x), 1.27x at S = 24-32 and 1.12x at S = 38-128. It loses only at S = 96-128 with
  K <= 4096 (to data-parallel and the owner) and, by 0.5-2%, to `atomic` at S = 96-128, K >= 8192.

**Small K** (K = 512 and 1024, S = 1-128):
- **K = 1024: `atomic_iz` is the fastest tail for S <= 32**, 1.03-1.78x shorter than data-parallel's
  (S = 1: 8.4 against 13.7 us, at the default, which is 8 per tile here). Reduce-scatter beats
  data-parallel only at S <= 8 (1.01-1.19x); the
  owner never does.
- **K = 512: nothing beats data-parallel** (7.4-12.9 us); `atomic_iz` ties it within 1-6% for S <= 8.
- The launch-zeroed `atomic` is host-bound at K = 512: the Python launcher cannot issue the zeroing
  and the GEMM fast enough, and 7-15 us pass between them. From K = 1024 the gap is 0. A graph or C++
  launcher would remove it; `atomic_iz` does not have it.

**Packing the active programs onto few XCDs** (`STREAMK_BALANCE_XCDS=0`: the first S n stream-K ids,
so a tile's programs share an XCD and the tail runs on ceil(S n / 32) XCDs; S = 1-8):
- **It helps the atomic endings only when the whole tail fits one or two XCDs:** S = 1 (`atomic_iz`
  7.9 to 6.5 us at K = 2048, 14.8 to 8.5 at K = 8192) and S = 2 at K <= 4096 (8-10%). All of a tile's
  adds then go to one L2.
- **It hurts everything else:** from S = 3 the atomic endings are 0-45% slower packed, and reduce-scatter
  is slower packed at every S except S = 1, K = 8192 (up to 2.1x at S = 4-8), because the tail's K
  loop and partial traffic run on fewer XCDs.
- Default spreading stays right; a placement rule would only pay for S <= 2 with the atomic ending.
- The default split (STREAMK_NUM_PROGRAMS unset) is this packed placement whenever K caps n: only the
  lowest stream-K ids get work, 32 to an XCD. At S = 1, K = 2048, 16 programs work, all on XCD 0. That is
  why the default looked fastest for the atomic ending at S = 1 (`atomic_iz`: 6.6 us default, 6.5 packed,
  7.9 spread).

**When the atomic ending is valid.** It loses and doubles nothing (writers on different XCDs sum exactly;
a doubled partial is rejected at about 700 times the error), but it changes the result:
- **Linear epilogues only:** the bias, once, from chunk 0; a scale alpha on every partial; beta C by
  initialising the tile to beta C instead of zero. Activations, clamps and output quantisation would need
  a second pass.
- **16-bit C only:** fp32 atomics are slower than storing the partials and reading them back.
- **Not deterministic:** fine for inference that tolerates bf16-level error; not for training
  reproducibility, golden-output tests or debugging.

**Zeroing alternatives weighed, not built.** `atomic_iz` was chosen because its only wait falls before the
K loop, where it is hidden. The others:
- chunk 0 stores and raises a flag, and the others add after it: no zeroing, but an exposed wait at the
  end;
- the first arriver on a counter stores: a counter round trip plus the same wait;
- the caller guarantees a zeroed C: free in the GEMM, but a contract on the caller;
- an fp32 self-cleaning workspace with a last arriver: one rounding, but fp32 atomics at 4x a store, and
  a counter.

## 3. Write-through partials with relaxed flags and barrier (2026-10-08)

The reduce-scatter and owner endings, v20's code, get v20 entry 14's change: write-through
partials and the drain in every wave as before, then relaxed `gpu`-scope flags, arrival,
generation bump and polls, with `s_waitcnt vmcnt(0)` between the generation read and the arrival
and a compiler-only fence (`asm_fence`) where the acquires were. This is recommendation 1 of
[memory_model_litmus](../../../../../experiments/memory_model_litmus/WRITEUP.md). The atomic
endings were relaxed already and are unchanged. No `buffer_wbl2` or `buffer_inv` is left in any
ending.

Old and new code on GPU 0 without the amdgcnas hook, two interleaved rounds
([results/sync_wt.md](results/sync_wt.md)):
- **Correct:** `check_streamk.py`, reduce-scatter and owner: 108 configurations and 36 with
  `--policies capped`, all right with flags and counts re-armed. The atomic ending passes its own
  check, `check_tile_limit.py --versions 22`: 1630 of 1630 cases.
- `check_streamk.py`'s default set also runs the atomic ending, and its plain `allclose` rejects
  it on most shapes, before this change as after (old code: 30, 10 and 0 wrong launches of 30 on
  three shapes). It rounds C once per program; entry 1's mean-error check is the one that applies.
- **Cold kernel time** under the default capped rule: -1.8, -1.3 and 0.0 us on 4352x4096x8192,
  4352x4352x8192 and 3328x5120x8192. **do_bench:** +2.3-2.9% at 4352x4096x4096 in both rounds.
