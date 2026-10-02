# Bias and VGPR spills in the gfx950 a16w16 persistent kernels

Compiled with [check_spills.py](check_spills.py), one fresh Triton cache per case, M=N=4096, bf16
inputs. Kernels are AITER's ports in
`aiter/ops/triton/_gluon_kernels/gfx950/gemm/basic/gemm_a16w16{,_persistent}.py` (v9 = `compute_bound`);
`tut_v10` is the tutorial's own `v10_persistant` with the same bias epilogue
([tutorial_v10_bias.py](tutorial_v10_bias.py)). Raw tables: [results/](results/).

Stacks:

- **m1**: Triton `gfx950-tutorial-v2.2`, AITER as shipped: AITER's llirSched plugin plus per-MFMA
  `cd_regclass="a"` pins, no fixed AGPR/VGPR split.
- **m2**: m1 plus `amdgpu-agpr-alloc=256` (the tutorial's fixed split).
- **m3**: Triton `gfx950-tutorial-v2.1`, the tutorial's `llir+force-agpr` (`TRITON_FORCE_MFMA_AGPR=1`:
  `amdgpu-agpr-alloc=256` + `amdgpu-mfma-vgpr-form=0`) with the tutorial's `libLlirSched.so`.
  AITER's kernels run with `cd_regclass` dropped.
- **m3a**: m3 plus amdgcnas.

m3 and m3a describe the tutorial before the v2.2 re-pin (#54). Since #54 the tutorial runs on
v2.2 with `cd_regclass="a"` in the kernels, which is m1's AGPR mechanism: the tutorial's `llir`
config is m1 with the tutorial's plugin, and `llir+amdgcnas` adds amdgcnas.

"Arch VGPRs" is `; NumVgprs`, capped at 256 on gfx950; the 256 AGPRs are the four fp32 accumulator
quadrants. Spill-site columns count `scratch_store`/`scratch_load` instructions: "tile loop" = inside
the persistent tile loop but outside the K loop, "outside" = before the tile loop. No case spills
inside the K loop.

## 1. Bias causes the spills (m1, AITER as shipped)

| Kernel | Out | Bias off: VGPRs / total / spills | Bias on: VGPRs / total / spills (scratch B) |
|---|---|---|---|
| v9 | bf16 | 190 / 448 / 0 | 238 / 496 / 0 |
| v9 | fp32 | 214 / 472 / 0 | 240 / 496 / 0 |
| **v10** | **bf16** | 224 / 480 / 0 | **256 / 512 / 6 (28)** |
| **v10** | **fp32** | 244 / 500 / 0 | **256 / 512 / 24 (100)** |
| v11 | bf16 | 194 / 452 / 0 | 242 / 500 / 0 |
| v11 | fp32 | 226 / 484 / 0 | 256 / 512 / 0 |
| v12 | bf16 | 184 / 440 / 0 | 224 / 480 / 0 |
| v12 | fp32 | 200 / 456 / 0 | 228 / 484 / 0 |
| v13 | bf16 | 222 / 480 / 0 | 234 / 492 / 0 |
| **v13** | **fp32** | 204 / 460 / 0 | **256 / 512 / 8 (36)** |

Identical at K=8192 and K=1024 ([results/baseline_m1.md](results/baseline_m1.md)). All outputs match
an fp32 `F.linear` reference.

- Confirmed: v10 spills only with bias, 6 VGPRs with bf16 output and 24 with fp32 output. v13 with
  fp32 output also spills (8), and v11 with fp32 output sits exactly at 512 without spilling.
- Bias costs 32-48 arch VGPRs. Two fp32 halves in `SliceLayout(0, MFMA)` are 32 VGPRs; `gl.load`
  on a pointer tensor adds 64-bit per-lane addresses on top.
- What spills: tile-invariant values that LLVM hoists out of the tile loop, for v10 bf16 the
  `stride_cm * offs_cm` row terms of the C store offsets. They are stored once before the tile loop
  and reloaded in every tile's epilogue, after the K loop. The K loop itself stays clean, so the
  cost is a handful of scratch reloads per tile, but it is a spill all the same, and it grows with
  fp32 output.

## 2. Does the AGPR mechanism matter? (v9, v10 at K=8192)

| Kernel | Out | Bias | m1 | m2 | m3 | m3a |
|---|---|---|---|---|---|---|
| v9 | bf16 | off | 448 / 0 | 448 / 0 | 488 / 0 | 488 / 0 |
| v9 | bf16 | on | 496 / 0 | 496 / 0 | 512 / 0 | 512 / 0 |
| v9 | fp32 | off | 472 / 0 | 472 / 0 | 472 / 0 | 472 / 0 |
| v9 | fp32 | on | 496 / 0 | 496 / 0 | 512 / **20** | 512 / **20** |
| v10 | bf16 | off | 480 / 0 | 480 / 0 | 488 / 0 | 488 / 0 |
| v10 | bf16 | on | 512 / **6** | 512 / **6** | 512 / **18** | 512 / **18** |
| v10 | fp32 | off | 500 / 0 | 500 / 0 | 472 / 0 | 472 / 0 |
| v10 | fp32 | on | 512 / **24** | 512 / **24** | 512 / **40** | 512 / **40** |

Cells are total VGPRs (arch + AGPR) / spilled VGPRs. `tut_v10` gives exactly the same numbers as
AITER's v10 under m3 and m3a, so the AITER port is faithful and the spill is not a porting artefact.

- **m1 = m2 in every case.** The fixed 256/256 split changes nothing: the accumulators fill the
  256 AGPRs either way, and arch VGPRs are capped at 256 regardless, so no split can give the
  kernel more VGPRs.
- **amdgcnas does not change spills** (m3 = m3a). It rewrites the assembly (4032 to 2523 lines for
  v10) and recomputes `.vgpr_count`, but register allocation has already happened.
- **The v2.2 + `cd_regclass` stack is mostly better than the tutorial's v2.1 force-agpr stack:**
  it saves 40 VGPRs on v9 (bf16, no bias) and 8 on v10, cuts the v10 bias spills from 18 to 6
  (bf16) and from 40 to 24 (fp32), and v9 fp32 with bias from 20 to 0. The one exception is v10
  with fp32 output and no bias (500 vs 472). The comparison changes the Triton pin as well as the
  AGPR mechanism, but m1 = m2 shows the fixed split is not what matters, so the difference comes
  from v2.2 and/or the per-MFMA pins replacing `amdgpu-mfma-vgpr-form=0`.
- Recommendation: switching the tutorial default to v2.2 + `cd_regclass="a"` is worth doing as a
  separate change (re-pin, `cd_regclass="a"` on every MFMA, drop `amdgpu-agpr-alloc`, port AITER's
  pin-aware llirSched changes, update `CONFIG_ENV`, refresh the perf tables). It does not by itself
  remove the v10 bias spill, so the kernel-side fix below is still needed. amdgcnas is independent
  of the switch as far as registers go (it cannot add or remove spills); keeping it is a
  performance question to settle on the refreshed perf tables.

## 3. Mitigations

All on m1 (AITER's stack), K=8192 and 1024 (identical results at both K).

**Option A, bias in the accumulator init (`_init_acc`).** The accumulators start as the broadcast
fp32 bias instead of zero, loaded at tile start with `buffer_load` (scalar base, 32-bit lane
offsets) ahead of the prologue async copies; the epilogue add goes away. Without bias
`_init_acc` returns the same constant zeros as before, and the no-bias `.amdgcn` of every kernel is
instruction-for-instruction identical to the baseline (only DWARF line numbers differ).

| Kernel | Out | Baseline bias: VGPRs / spills | Option A bias: VGPRs / spills |
|---|---|---|---|
| v10 | bf16 | 256 / 6 | **232 / 0** |
| v10 | fp32 | 256 / 24 | **246 / 0** |
| v11 | bf16 | 242 / 0 | 208 / 0 |
| v11 | fp32 | 256 / 0 | 240 / 0 |
| v12 | bf16 | 224 / 0 | 188 / 0 |
| v12 | fp32 | 228 / 0 | 203 / 0 |
| v13 | bf16 | 234 / 0 | 230 / 0 |
| v13 | fp32 | 256 / 8 | **216 / 0** |

Bias then costs 2-14 arch VGPRs instead of 32-48. On m3 (v2.1 force-agpr) Option A also takes
v10-v13 with bias to 0 spills ([results/opt_a_m3.md](results/opt_a_m3.md)).

**Option B, split and delayed epilogue loads** (v10 only: `bias_l` loaded one MFMA before the `tl`
store, `bias_r` after the `bl` store, bf16 until the add, `buffer_load`): bf16 output 256 VGPRs /
0 spills (at the ceiling), fp32 output 256 / 14 spills ([results/opt_b_v10.md](results/opt_b_v10.md)).
Strictly worse than A, so dropped. Option C (LDS-staged bias, store-layout bias) was not needed.

**Where Option A is applied.** With Option A everywhere, v11 and v12 (which never spilled) measured
about 1% slower with bias on few-tile shapes ([results/perf_bias_opt_a_all.csv](results/perf_bias_opt_a_all.csv),
rechecked with 9 rounds in [results/perf_bias_recheck.csv](results/perf_bias_recheck.csv)),
while v10 gained and v13 was neutral. So the shipped change applies it only where the epilogue bias
spills:

- v10: bias in the accumulator init.
- v13: `BIAS_IN_ACC=True` on the shared v12/v13 kernel.
- v11: unchanged.
- v12: `BIAS_IN_ACC=False`, same code as before.

Final spill check ([results/final_m1.md](results/final_m1.md)): no spills in any v9-v13 case on
m1. On m3 v11 and v12 still spill with the epilogue bias (v11: 19 bf16 / 49 fp32, v12: 27 fp32;
[results/final_m3.md](results/final_m3.md)); that only matters for a bias port into the tutorial on
v2.1, where Option A in all four kernels removes them.

## 4. Validation

- AITER tests: `op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py -k gluon_compute_bound`, 260
  passed (v9-v13, bias on/off, fp16/bf16, with/without a preallocated output, fp32 output, shapes
  with fewer tiles than CUs and uneven tiles per program). Run with
  [run_aiter_tests.py](run_aiter_tests.py), which bootstraps this environment (no `flydsl`, pip ROCm).
- Performance, [perf_bias.py](perf_bias.py): 12 shapes from `lixun_aiter_losses.csv` with bias
  switched on (the CSV itself has no bias shapes), `do_bench_cudagraph`, 5 interleaved rounds,
  median, GPU 3, "orig" = [orig/gemm_a16w16_persistent_orig.py](orig/gemm_a16w16_persistent_orig.py).
  `run_aiter_compare.py` needs the full `aiter` import (AITER's tuned pick), which this environment
  cannot provide. TFLOPS, no bias / bias orig / bias new (new vs orig):

| Shape (tiles) | v10 | v13 | v11 (code unchanged) | v12 (code unchanged) |
|---|---|---|---|---|
| 2048x2304x16384 (72) | 635 / 633 / 630 (-0.5%) | 635 / 634 / 636 (+0.4%) | 636 / 637 / 637 (-0.0%) | 632 / 633 / 632 (-0.1%) |
| 16384x512x8192 (128) | 996 / 976 / 986 (+1.0%) | 994 / 985 / 986 (+0.1%) | 999 / 984 / 982 (-0.2%) | 990 / 987 / 984 (-0.4%) |
| 16384x1024x7168 (256) | 1428 / 1420 / 1421 (+0.1%) | 1433 / 1424 / 1427 (+0.2%) | 1431 / 1425 / 1422 (-0.2%) | 1428 / 1425 / 1426 (+0.1%) |
| 4096x8192x1024 (512) | 1162 / 1138 / 1147 (+0.8%) | 1181 / 1162 / 1156 (-0.6%) | 1200 / 1174 / 1174 (+0.0%) | 1199 / 1181 / 1179 (-0.2%) |
| 8192x5120x25600 (640) | 1381 / 1378 / 1377 (-0.1%) | 1385 / 1385 / 1378 (-0.5%) | 1384 / 1383 / 1383 (-0.0%) | 1385 / 1382 / 1382 (+0.0%) |
| 8192x8192x8192 (1024) | 1484 / 1474 / 1476 (+0.1%) | 1488 / 1480 / 1481 (+0.0%) | 1488 / 1484 / 1480 (-0.3%) | 1489 / 1484 / 1484 (+0.0%) |
| 10240x8448x7168 (1320) | 1350 / 1339 / 1344 (+0.4%) | 1355 / 1351 / 1351 (-0.0%) | 1357 / 1350 / 1350 (+0.0%) | 1359 / 1352 / 1352 (+0.0%) |
| 16384x8192x1024 (2048) | 1238 / 1203 / 1222 (+1.6%) | 1267 / 1246 / 1239 (-0.5%) | 1283 / 1247 / 1248 (+0.1%) | 1280 / 1258 / 1259 (+0.1%) |
| 32768x5120x640 (2560) | 1076 / 1037 / 1067 (+2.9%) | 1124 / 1092 / 1095 (+0.3%) | 1137 / 1096 / 1089 (-0.6%) | 1142 / 1111 / 1113 (+0.2%) |
| 32768x7168x1792 (3584) | 1377 / 1349 / 1365 (+1.2%) | 1396 / 1380 / 1379 (-0.1%) | 1403 / 1379 / 1377 (-0.2%) | 1403 / 1386 / 1386 (+0.0%) |
| 16384x16384x4096 (4096) | 1483 / 1463 / 1477 (+1.0%) | 1499 / 1493 / 1502 (+0.6%) | 1494 / 1487 / 1470 (-1.2%) | 1495 / 1489 / 1490 (+0.1%) |
| 32768x8192x7168 (4096) | 1570 / 1561 / 1566 (+0.3%) | 1584 / 1575 / 1575 (-0.0%) | 1587 / 1576 / 1576 (-0.0%) | 1588 / 1576 / 1575 (-0.1%) |

v11 and v12 compile to the same code as before, so their columns show the run-to-run noise
(up to about 1.2%). Against that, v10 with bias gains 0.8-2.9% on the small-K shapes, where the
epilogue is a larger share of the tile, and is neutral elsewhere; v13 is neutral. v10's bias cost
drops from up to 3.6% to up to 1.3% (32768x5120x640: 1076 without bias, 1037 before, 1067 after).
All bias outputs match the fp32 reference.
