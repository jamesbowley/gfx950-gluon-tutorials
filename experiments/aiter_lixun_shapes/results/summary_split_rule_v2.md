# v20 stream-K policies on the 630 shapes (v20 only)

Shapes: `supported_status_scoped.csv`, timed as in the AITER comparison (`do_bench_cudagraph`, 2 interleaved rounds, median) but without the AITER worker (`sweep --v20-only`).

- Stack: torch 2.13.0+rocm7.14.0, Triton 3.8.0 @ /home/jbowley/repos/triton_gfx950-tutorial-v2.2/python/triton.
- GPU: HIP device 0, PCI bus 117, gfx950:sramecc+:xnack-.
- v20: `v20_streamk_reduce_scatter` (snapshot sha256 926bc1ffe0edf0bb), v9 tile order.

Measured 630 shapes.

## Diagnostic: auto's stream-K choice against forcing data-parallel

On the 356 shapes where auto picks a stream-K path. Positive means auto's choice is faster than v20 `dp`; this column never counts as v20's result.

| | shapes | auto faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| all | 356 | 292 | 8 | +19.8% | -15.3% | +104.4% |
| one-tile owner, K < 2048 | 42 | 0 | 0 | -24.2% | -43.7% | -8.7% |
| one-tile owner, K >= 2048 | 230 | 216 | 7 | +17.9% | +1.8% | +71.6% |
| one-tile reduce-scatter, K < 2048 | 7 | 0 | 0 | -23.9% | -24.5% | -21.6% |
| one-tile reduce-scatter, K >= 2048 | 68 | 67 | 0 | +99.5% | +44.8% | +150.1% |
| two-tile (rs), K >= 2048 | 9 | 9 | 1 | +11.2% | +4.8% | +25.7% |

## v20 capped against v20 auto

`STREAMK_POLICY=capped` against the default rule. Where capped picks data-parallel its time is the data-parallel column (the same kernel). Positive means capped is faster.

| | shapes | capped faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| all shapes | 630 | 289 | 425 | +0.0% | -0.0% | +23.6% |
| under 1 wave | 321 | 215 | 148 | +1.7% | -0.1% | +36.5% |
| 1 to 2 waves | 84 | 23 | 64 | +0.0% | -0.3% | +4.7% |
| 2 to 4 waves | 68 | 27 | 59 | +0.0% | +0.0% | +1.0% |
| 4 to 8 waves | 67 | 16 | 64 | +0.0% | +0.0% | +0.4% |
| 8 waves or more | 90 | 8 | 90 | +0.0% | +0.0% | +0.0% |

### By auto's pick and capped's pick

| | shapes | capped faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| auto data-parallel, capped data-parallel | 272 | 0 | 272 | +0.0% | +0.0% | +0.0% |
| auto data-parallel, capped spread | 2 | 2 | 1 | +2.3% | +0.4% | +4.1% |
| auto one-tile owner, capped data-parallel | 51 | 48 | 0 | +23.5% | +1.7% | +75.7% |
| auto one-tile owner, capped one-tile owner | 221 | 162 | 146 | +0.3% | -0.3% | +10.4% |
| auto one-tile reduce-scatter, capped data-parallel | 1 | 1 | 0 | +70.4% | +70.4% | +70.4% |
| auto one-tile reduce-scatter, capped one-tile rs | 74 | 71 | 2 | +20.9% | +1.8% | +45.7% |
| auto two-tile (rs), capped one-tile rs | 9 | 5 | 4 | +0.6% | -0.7% | +2.0% |

## Wrong outputs, failures and notes

None: every v20 output matched the reference, v20 matched again after timing, and its flags and barrier counts were all back at 0.
