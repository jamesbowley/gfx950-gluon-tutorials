# E6: counters while running back-to-back

**SPM is not usable for this.** `rocprofv3 --spm-beta-enabled --spm GRBM_GUI_ACTIVE GRBM_COUNT
--spm-sample-interval 10000` runs SPM as a per-dispatch collection mode
(`ROCPROF_SPM_COUNTER_COLLECTION`): under it a back-to-back GEMM took 4570 us instead of
29 us, the run wrote no output files, and the tool refuses to combine SPM with `--pmc`. So
neither SPM nor `--pmc` can observe kernels running back-to-back.

**Fallback: kernel trace.** `rocprofv3 --kernel-trace` does not serialise graph launches. Under
it, `place.py sweep --what c --start 32 --stop 160 --step 128 --fixed 900` timed C at 32 MB
(fast half) and then 160 MB (slow half) of one 1 GB buffer, 4401 GEMMs each, and all 8802
kernels were traced. Statistics over the last three quarters of each segment:

| C | graph time per GEMM (us) | kernel duration, median (p10 - p90) | gap to the next kernel |
|---|---|---|---|
| 32 MB, fast | 29.25 | 28.56 (27.96 - 29.48) | 0.00 |
| 160 MB, slow | 30.07 | 29.36 (29.08 - 29.88) | 0.00 |

Duration histograms (0.25 us bins, counts above 20):
- fast: 27.75: 180, 28.00: 534, 28.25: 509, 28.50: 589, 28.75: 508, 29.00: 387, 29.25: 239,
  29.50: 202, 29.75: 73, 30.00 - 30.75: 160
- slow: 29.00: 543, 29.25: 1223, 29.50: 679, 29.75: 536, 30.00: 251, 30.25: 127, 30.50: 23

Duration hardly depends on the position within the 400-call graph (the first calls of each
replay are no different).

**Conclusion.** The slow half's cost is inside every kernel's execution, not between kernels.
Tracing adds about 0.7 us per kernel but keeps the 0.8 us difference. Fast kernels spread
widely (about 28.0 - 29.5 us); slow kernels sit in a narrower band 0.8 us higher. That fits a
GPU dithering at the power limit in the fast case (E1: 1390 W) and held below it in the slow
case (1335-1340 W, lower clock). Locking the clock (E5) is still the cleanest way to split the
cost into clock and cycles; it needs a root shell.
