# E5: locking the clock (blocked)

`bash e5_clock_lock.sh 1800` was meant to cap GPU 2's gfx clock with performance determinism
and rerun E1 and a coarse E2 sweep. In this container it cannot: `amd-smi set` fails with
`AmdSmiPermissionDeniedException ... requires elevation`, `/sys` is mounted read-only
(`power_dpm_force_performance_level`, `pp_od_clk_voltage`, `pp_dpm_sclk` and `power1_cap` are
all read-only), and the GPU stayed in `PERF_LEVEL: AUTO`. The script's exit trap reset the
setting anyway. To run it, use a host shell with root, or a container with a writable `/sys`.

## Fallback: cycles from E1's clock samples

E1 sampled hwmon sclk while each pattern replayed. Time x clock estimates the cycles per GEMM:

| buffer | us | MHz | cycles per GEMM |
|---|---|---|---|
| F1 (seed 0) | 28.33 | 2046 | 57.96k |
| F2 (seed 0) | 28.43 | 2039 | 57.97k |
| S1 (seed 0) | 29.45 | 2001 | 58.93k |
| S2 (seed 0) | 29.44 | 2004 | 59.00k |
| F1 (seed 1) | 28.52 | 2034 | 58.01k |
| S1 (seed 1) | 29.49 | 2011 | 59.30k |

Slow is about 3.8% longer in time, 1.9% lower in clock and about 1.9% more in cycles. Both
halves are at the board's power limit or close to it (1390 W fast, 1335-1340 W slow), yet slow
draws less power at a lower clock, so a power cap alone does not set slow's clock. hwmon
reports a firmware average, so this split is an estimate; E6 measures cycles directly.
