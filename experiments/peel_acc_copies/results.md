# Why v13 has 324-348 accumulator copies, and v12 has none

a16w16 v13 (`persistant_peel_acc`), and v14/v20, which compile to the same code at one tile per
program, carry 348 `v_accvgpr_mov_b32` with the published stack (324 without the plugins). v12
has none. All of them are outside the K loop: 300 in the two-step drain and 48 where the peeled
K-steps hand over to the loop. At 4096x4096x1024 they made v13's waves about 6% longer than v9's
(about 3k cycles per tile, fixed, so about 1% at K=8192).

**Cause.** The copies are a register-allocation side effect of how Triton v2.2 keeps MFMA
accumulators in AGPRs. The peel did not introduce them directly; it changed the
interval structure enough that LLVM's greedy allocator stopped packing it.

1. `cd_regclass="a"` is an empty `"=a,0"` inline asm around each MFMA tile, and the backend
   selects every MFMA in **VGPR form** (`V_MFMA_..._vgprcd_e64`). Before allocation each MFMA
   works on a short VGPR temporary, copied in from and out to the pinned AGPR accumulator.
2. `amdgpu-rewrite-agpr-copy-mfma` runs after greedy has assigned registers. It turns each MFMA
   chain into AGPR form and moves it onto the AGPR the result was being copied to. When an
   accumulator stays in one AGPR group, every copy becomes an identity and disappears.
3. The kernels use all 256 AGPRs (64 accumulator tiles x 4) and peak at exactly 64 live tiles
   in the drain, so the allocation has zero slack. In v13, greedy fails to pack the drain: it
   local-splits 16 accumulator intervals out to VGPRs between K-steps 14 and 15 and reuses their
   AGPRs (`a[192:195]`, `a[0:3]`, `a[92:95]`) as shared slots for other accumulators. After the
   rewrite pass, those chains sit in a shared slot rather than their own registers, which
   leaves 81 real AGPR-to-AGPR copies: 81 x 4 = 324 `v_accvgpr_mov`.

**Fix.** Selecting MFMAs in AGPR form up front with `-amdgpu-mfma-vgpr-form=0` (what the v2.1
stack's `TRITON_FORCE_MFMA_AGPR=1` set) removes every copy in v9-v13, with and without bias, at
any K. In warm thread traces it brings v13's waves from 49,324 to 47,142 cycles (-4.4%), level
with v9. Stock Triton v2.2 cannot set it (see the recommendation).

## Tooling

| Script | What it does |
|---|---|
| [compile_stats.py](compile_stats.py) | Builds a version or variant in a fresh cache; counts copies (before, in, after the K loop), `s_nop`, accumulator reads and writes, registers, spills; `--keep` saves `.llir`/`.amdgcn` |
| [mir_copies.py](mir_copies.py), [mir_section.py](mir_section.py) | AGPR copies after every machine pass of an `LLVM_IR_ENABLE_DUMP=1` dump (Triton's in-process `print-after-all`); extract chosen passes' MIR |
| [llc_variants.py](llc_variants.py) | Copies under `llc` option variants on a saved `.llir` |
| [llc_stage.py](llc_stage.py), [bench_llc.py](bench_llc.py) | Stages hook: the amdgcn stage built by the pinned `llc` with `PAC_LLC_FLAGS`, then amdgcnas; `bench.py` with that hook |
| [time_fix.py](time_fix.py) | Cold rocprof time plus copies per build, interleaved rounds |
| [att_phases.py](att_phases.py), [att_ksteps.py](att_ksteps.py) | Thread-trace phases (prologue, MFMA span, epilogue) and MFMA bubbles per K-step, on `rocprofv3 --att` output |

Environment as in the a16w16 benchmarks: `PYTHONPATH=<repo>/scripts/triton_deepbind_shim:<triton_gfx950-tutorial-v2.2>/python`,
`HIP_VISIBLE_DEVICES=2`. Compile studies ran without plugins unless noted (they do not cause the
copies). Timing and traces used the standard `llir+amdgcnas` stack. All at 4096x4096xK BF16, MI355X.

## 1. Which builds have copies

[results/copies_inprocess.md](results/copies_inprocess.md): Triton's in-process codegen.

| version | copies | with plugins | with bias | greedy local splits (llc) | VGPRs |
|---|---|---|---|---|---|
| v9  | 128 | 128 | 188 | 16 | 448 |
| v10 | 128 | 128 | 152 | 16 | 480 |
| v11 | 0   | 0   | 132 | 0  | 452 |
| v12 | 0   | 0   | 0   | 0  | 440 |
| v13 | 324 | 348 | 356 | 32 | 484 |

The counts do not depend on K (1024 and 8192 are identical). A build has copies exactly when greedy
splits accumulator intervals, and small source changes flip it either way: bias gives v11 132
copies, v12 stays at 0.

## 2. Standalone reproduction

`llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -O3 -fp-contract=fast -relocation-model=pic` on the
cached `.llir` gives v9 128, v10 128, v11 0, v12 0 and v13 312 copies (Triton in-process: 324). The
`separate-const-offset-from-gep,early-cse` pre-pass makes no difference, and Triton runs it with
`canonicalize_gep=False` anyway. With the plugins, the `llc` stage hook matches Triton exactly (v13: 348).

## 3. Where the copies come from (MIR)

AGPR-destination `COPY`s after each machine pass, from Triton's in-process `print-after-all`
(no plugins). Greedy's own `lr-split` copies are not in these counts.

| after pass | v12 | v13 |
|---|---|---|
| two-address | 896 (576 AGPR-to-AGPR) | 1280 (832) |
| register coalescer | 512 (0) | 384 (0) |
| greedy, VGPR/AGPR run | 512 | 384 + 32 `lr-split` copies |
| amdgpu-rewrite-agpr-copy-mfma | 768 (512 AGPR-to-AGPR) | 704 (704) |
| virtregrewriter (identity copies deleted) | 256 (0 AGPR-to-AGPR) | 106 (106) |
| machine-cp | 256 (0) | 81 (81) |
| final `v_accvgpr_mov` | 0 | 324 |

- **MFMA forms:** after greedy, every MFMA is `V_MFMA_F32_16X16X32_BF16_vgprcd_e64` (512 in v12, 768 in v13). After the rewrite pass every one is `_e64` (AGPR form). The rewrite pass's debug log shows each chain's VGPR temporaries packed into a few physical groups (`$vgpr0_3`, `$vgpr38_41`, ...) and then reassigned to AGPRs. No reassignment fails.
- **v12's 256 remaining COPYs** are its zero initialisation (`v_accvgpr_write`). v13 has none, because the peel's zero-C MFMAs replace it.
- **The split, in v13's final MIR:**

  ```
  $vgpr150_vgpr151_vgpr152_vgpr153 = lr-split COPY killed $agpr192_agpr193_agpr194_agpr195
  $agpr192_agpr193_agpr194_agpr195 = COPY killed $agpr196_agpr197_agpr198_agpr199
  $agpr192_agpr193_agpr194_agpr195 = V_MFMA_F32_16X16X32_BF16_e64 ..., killed $agpr192_agpr193_agpr194_agpr195
  ```

  All 32 `lr-split` copies are in the drain block. 16 move accumulators out to VGPRs at K-step 14 (MFMAs 34-128 of the block), and 16 bring them back at K-step 15 (MFMAs 160-254).

**Greedy (llc `-debug-only=regalloc`).**

| | v12 | v13 |
|---|---|---|
| `AReg_128` intervals before allocation | 256 | 320 |
| max simultaneously live | 64 | 64 |
| evictions | 0 | 154 |
| local splits | 0 | 32 |

- **Intervals per accumulator in v12:** one long interval across the tile (header, K loop and drain), one inside the loop, and two inside the drain.
- **v13** adds 64 intervals for the peeled K-steps in the tile-loop header (median 1,608 instructions long), and its long interval grows from 8.9k to 12.1k instructions.
- **Not a pressure problem:** both versions peak at exactly 64 live tiles, the size of the AGPR file, so packing is possible in principle. No misaligned tuples (`AReg_128_Align2` starting at an offset that isn't a multiple of 4) were assigned in either version, so this isn't fragmentation. The scheduler's tracker gives the same drain pressure for both (204 VGPRs, 256 AGPRs), and it reverts its schedule in every region of both, so the instruction order is the source order. Greedy allocates in priority order, not program order, and with zero slack v13's extra intervals make it evict and then split.

**Hypothesis ruled out:** a zero-trip loop bypass merging the peel results into the drain. The tile-loop header (`bb.2`) has the K loop (`bb.3`) as its only successor, so there is no such phi.

## 4. Compiler options (llc on v13's `.llir`)

[results/llc_options.md](results/llc_options.md)

| option | v13 copies | v12 copies | v10 copies | v13 VGPRs |
|---|---|---|---|---|
| baseline | 312 | 0 | 128 | 496 |
| `-enable-local-reassign`, `-exhaustive-register-search`, `-lcr-max-depth=20 -lcr-max-interf=64`, `-split-spill-mode=size/speed` | 312 | 0 | 128 | 496 |
| `-greedy-regclass-priority-trumps-globalness` | 312 | 0 | 128 | 500 |
| `-greedy-reverse-local-assignment` | 496 | 0 | 0 | 492 |
| `-enable-misched=0` | 424 | 0 | 0 | 472 |
| `amdgpu-agpr-alloc=256` attribute only | 312 | 0 | - | 496 |
| **`-amdgpu-mfma-vgpr-form=0`** | **0** | 0 | 0 | 480 |

`selectAGPRFormMFMA()` (`SIMachineFunctionInfo.h`) needs `!MFMAVGPRForm && MinNumAGPRs >= NumRegs`.
The option defaults to true, so the attribute alone changes nothing.

## 5. `-amdgpu-mfma-vgpr-form=0` on every version

[results/llc_mfma_vgpr_form.md](results/llc_mfma_vgpr_form.md), llc, no plugins.

| build | copies before / after | VGPRs before / after | spills after |
|---|---|---|---|
| v9  | 128 / 0 | 448 / 440 | 0 |
| v10 | 128 / 0 | 476 / 452 | 0 |
| v11 | 0 / 0 | 452 / 500 | 0 |
| v12 | 0 / 0 | 440 / 468 | 0 |
| v13 | 312 / 0 | 496 / 480 | 0 |
| v9 bias  | 188 / 0 | 460 / 460 | 0 |
| v10 bias | 152 / 0 | 488 / 472 | 0 |
| v11 bias | 132 / 0 | 464 / 496 | 0 |
| v12 bias | 0 / 0 | 444 / 476 | 0 |
| v13 bias | 276 / 0 | 508 / 512 | **15** |

Adding `amdgpu-agpr-alloc=256` (the full v2.1 configuration) gives the same numbers, including
v13's bias spills.

**Kernel-side alternative tried:** leaving the peeled MFMAs unpinned (`cd_regclass=None`,
[variants](variants/), generated from v13) gives 96 copies and 14 spills (23 with bias). Worse.

## 6. Validation (standard plugin stack; both builds through the llc stage)

**Thread traces** ([results/att.md](results/att.md)): median cycles per wave over 8 waves of one
dispatch at K=1024. Warm traces are reproducible to about 0.5%. Cold ones vary by about 5% between
dispatches, so cold v9 is shown over three repeats.

| build | wave cycles | MFMA span | epilogue | MFMA bubbles k0-k1 / k14-k15 |
|---|---|---|---|---|
| v9 warm | 47,112 | 38,604 | 3,602 | 554 / 5,490 |
| v9 warm, flag | 46,130 (-2.1%) | 37,628 | 3,706 | 586 / 4,658 |
| v12 warm | 48,178 | 38,650 | 3,634 | 610 / 5,644 |
| v12 warm, flag | 47,900 (-0.6%) | 37,568 | 4,390 | 598 / 4,624 |
| v13 warm | 49,324 | 41,588 | 3,414 | 1,398 / 7,756 |
| v13 warm, flag | **47,142 (-4.4%)** | 38,538 | 4,302 | 920 / 5,206 |
| v13 cold / flag | 50,964 / 49,910 | 41,526 / 39,316 | 3,390 / 4,510 | 1,422 / 7,656 → 972 / 5,240 |
| v9 cold, 3 repeats | 49,624 / 50,326 / 49,034 | | | |
| v9 cold flag, 3 repeats | 54,136 (outlier) / 48,858 / 49,050 | | | |

With the flag, v13's MFMA span loses the 3k cycles it had over v12, and the drain bubbles fall
below v9's. Part of the gain is paid back in the epilogue (about +900 cycles), where the final
accumulator reads now sit.

**Cold kernel time** ([results/timing_raw.md](results/timing_raw.md)): rocprofv3, 1000 launches over
a rotating 512 MB, median of 3 interleaved rounds in us (min-max). Runs about 2 us slow (about 7%)
hit random rounds of any build, so this resolves only the larger changes.

| K | version | in-process | llc | llc + flag |
|---|---|---|---|---|
| 1024 | v9 | 30.46 (30.40-30.56) | 30.64 (30.44-31.84) | 32.36 (30.74-33.16) |
| 1024 | v10 | 30.96 | 31.08 | 30.80 |
| 1024 | v11 | 30.92 | 30.92 | 31.00 |
| 1024 | v12 | 30.96 | 31.08 | 31.28 (31.00-33.56) |
| 1024 | v13 | 31.24 (31.24-31.44) | 31.56 (31.24-33.54) | **30.82** (30.72-32.68) |
| 8192 | v9 | 175.30 | 175.14 | 175.60 |
| 8192 | v10 | 175.22 | 175.56 | 175.92 |
| 8192 | v11 | 175.54 | 175.56 | 175.84 |
| 8192 | v12 | 176.60 | 176.40 | 176.32 |
| 8192 | v13 | 176.90 | 177.10 | **175.74** (one round 200.76) |

- **v13 with the flag:** 2.3% faster than its llc baseline at K=1024 and 0.8% at K=8192, consistent with a fixed per-tile saving. This is the only version where the change stands out from the noise.
- **v9 at K=1024:** two of its three flagged rounds were slow, but its traces show no cost (46,130 against 47,112 warm, and level cold), so I read those rounds as outliers rather than a regression.
- **Everything else:** within noise. All builds pass `bench.py`'s correctness check.

## 7. Recommendation

1. **Build the a16w16 kernels with AGPR-form MFMA selection.** This needs Triton v2.2 to be able
   to set `amdgpu-mfma-vgpr-form=0`. Today `get_llvm_flags` and `DISABLE_LLVM_OPT` can only set
   options to `true`, `amdgpu-mfma-vgpr-form` is process-wide, and `llvm_fn_attrs` /
   `amdgpu-agpr-alloc` alone do nothing. A `name=value` form in the flags list, or bringing back
   v2.1's `TRITON_FORCE_MFMA_AGPR`, would do it; [llc_stage.py](llc_stage.py) does the same today
   through an external `llc`. Make it a per-kernel choice, not a global one:
   - It is clearly a win for v13/v14/v20, and v9 and v10 lose their 128 copies too.
   - v11 and v12 gain VGPRs (452 to 500 and 440 to 468) for no measured benefit.
   - v13 with bias spills 15 VGPRs, so the bias variant needs its own check.
2. **Keep the peel.** The peel itself works: it removes the zero-init writes and the steady-state K-steps are unchanged. The copies come from an allocation that has no slack in the drain, and any source change can tip it (bias gives v11 132 copies). Kernel-side workarounds are a lottery, and unpinning the peel is worse.
3. **Report upstream (LLVM AMDGPU).** With a full AGPR file, greedy splits accumulator intervals in a straight-line drain, and `amdgpu-rewrite-agpr-copy-mfma` (which has a TODO about loop phis) cannot keep the split chains in their own registers. `builds/v13.llir` plus the `llc` line above reproduces it at the pin `b010a18d`.
4. **Regression guard.** `regress.py` already parses every `.amdgcn` (`check_spills.parse_amdgcn`). Add a `v_accvgpr_mov` count next to the VGPR and spill columns and flag any nonzero value for v9 and later. This would have caught v13 when it landed: the loop-only MFMA efficiency at K=8192 could not.

## Reproduce

```bash
cd experiments/peel_acc_copies
python compile_stats.py 9 10 11 12 13 --keep builds            # section 1
python llc_variants.py builds/v13.llir builds/v12.llir          # section 4
LLC=/root/.triton/llvm/llvm-b010a18d-ubuntu-x64-1/bin/llc
$LLC -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -O3 -fp-contract=fast -relocation-model=pic \
    builds/v13.llir -o /dev/null -debug-only=regalloc 2> mir/v13_regalloc_debug.txt   # section 3
export LLVM_PASS_PLUGIN_PATH=<repo>/plugins/llir_scheduler/libLlirSched.so TRITON_AMDGCNAS_PLUGIN=1
python time_fix.py --versions 9,10,11,12,13 --K 1024,8192 --rounds 3   # section 6
```

The `print-after-all` dumps (about 200 MB per version), the scheduler and regalloc logs, and the raw
thread traces were deleted after analysis. The commands above regenerate them; `builds/`, `llc/`,
`mir/`, `att/` and `variants/` are git-ignored scratch.
