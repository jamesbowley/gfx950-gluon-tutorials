# v20 stream-K policies on the 630 shapes (v20 only)

Shapes: `supported_status_scoped.csv`, timed as in the AITER comparison (`do_bench_cudagraph`, 2 interleaved rounds, median) but without the AITER worker (`sweep --v20-only`).

- Stack: torch 2.13.0+rocm7.14.0, Triton 3.8.0 @ /home/jbowley/repos/triton_gfx950-tutorial-v2.2/python/triton.
- GPU: HIP device 0, PCI bus 117, gfx950:sramecc+:xnack-.
- v20: `v20_streamk_reduce_scatter` (snapshot sha256 5c3ca01e86c9eb53), v9 tile order.

Measured 630 shapes.

## Diagnostic: auto's stream-K choice against forcing data-parallel

On the 356 shapes where auto picks a stream-K path. Positive means auto's choice is faster than v20 `dp`; this column never counts as v20's result.

| | shapes | auto faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| all | 356 | 289 | 10 | +19.6% | -14.1% | +99.6% |
| one-tile owner, K < 2048 | 42 | 0 | 0 | -21.8% | -41.1% | -8.5% |
| one-tile owner, K >= 2048 | 230 | 213 | 9 | +17.6% | +2.1% | +72.8% |
| one-tile reduce-scatter, K < 2048 | 7 | 0 | 0 | -23.9% | -25.5% | -21.8% |
| one-tile reduce-scatter, K >= 2048 | 68 | 67 | 0 | +98.2% | +43.6% | +142.8% |
| two-tile (rs), K >= 2048 | 9 | 9 | 1 | +12.9% | +6.2% | +27.2% |

## v20 capped against v20 auto

`STREAMK_POLICY=capped` against the default rule. Where capped picks data-parallel its time is the data-parallel column (the same kernel). Positive means capped is faster.

| | shapes | capped faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| all shapes | 630 | 292 | 429 | +0.0% | -0.0% | +18.0% |
| under 1 wave | 321 | 214 | 154 | +0.8% | -0.1% | +27.0% |
| 1 to 2 waves | 84 | 27 | 63 | +0.0% | -0.2% | +5.2% |
| 2 to 4 waves | 68 | 27 | 59 | +0.0% | +0.0% | +0.7% |
| 4 to 8 waves | 67 | 16 | 63 | +0.0% | +0.0% | +0.3% |
| 8 waves or more | 90 | 8 | 90 | +0.0% | +0.0% | +0.0% |

### By auto's pick and capped's pick

| | shapes | capped faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| auto data-parallel, capped data-parallel | 272 | 0 | 272 | +0.0% | +0.0% | +0.0% |
| auto data-parallel, capped spread | 2 | 2 | 1 | +2.4% | +0.6% | +4.2% |
| auto one-tile owner, capped data-parallel | 51 | 49 | 2 | +21.0% | +2.2% | +69.0% |
| auto one-tile owner, capped one-tile owner | 221 | 170 | 149 | +0.3% | -0.2% | +9.3% |
| auto one-tile reduce-scatter, capped data-parallel | 1 | 1 | 0 | +78.0% | +78.0% | +78.0% |
| auto one-tile reduce-scatter, capped one-tile rs | 74 | 65 | 1 | +16.0% | -1.1% | +33.3% |
| auto two-tile (rs), capped one-tile rs | 9 | 5 | 4 | +0.3% | -2.4% | +0.7% |

## Wrong outputs, failures and notes

None: every v20 output matched the reference, v20 matched again after timing, and its flags and barrier counts were all back at 0.
