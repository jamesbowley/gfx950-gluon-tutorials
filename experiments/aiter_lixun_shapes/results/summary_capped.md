# v20 stream-K policies on the 630 shapes (v20 only)

Shapes: `supported_status_scoped.csv`, timed as in the AITER comparison (`do_bench_cudagraph`, 2 interleaved rounds, median) but without the AITER worker (`sweep --v20-only`).

- Stack: torch 2.13.0+rocm7.14.0, Triton 3.8.0 @ /home/jbowley/repos/triton_gfx950-tutorial-v2.2/python/triton.
- GPU: HIP device 0, PCI bus 117, gfx950:sramecc+:xnack-.
- v20: `v20_streamk_reduce_scatter` (snapshot sha256 93847ce6674f120e), v9 tile order.

Measured 630 shapes.

## Diagnostic: auto's stream-K choice against forcing data-parallel

On the 356 shapes where auto picks a stream-K path. Positive means auto's choice is faster than v20 `dp`; this column never counts as v20's result.

| | shapes | auto faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| all | 356 | 291 | 9 | +19.8% | -13.5% | +97.7% |
| one-tile owner, K < 2048 | 42 | 0 | 0 | -22.0% | -41.0% | -7.8% |
| one-tile owner, K >= 2048 | 230 | 215 | 8 | +17.9% | +2.0% | +72.0% |
| one-tile reduce-scatter, K < 2048 | 7 | 0 | 0 | -25.3% | -25.9% | -22.7% |
| one-tile reduce-scatter, K >= 2048 | 68 | 67 | 0 | +96.0% | +41.3% | +137.9% |
| two-tile (rs), K >= 2048 | 9 | 9 | 1 | +13.0% | +6.1% | +26.2% |

## v20 capped against v20 auto

`STREAMK_POLICY=capped` against the default rule. Where capped picks data-parallel its time is the data-parallel column (the same kernel). Positive means capped is faster.

| | shapes | capped faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| all shapes | 630 | 278 | 422 | +0.0% | -0.1% | +18.2% |
| under 1 wave | 321 | 209 | 151 | +0.9% | -0.3% | +27.4% |
| 1 to 2 waves | 84 | 26 | 61 | +0.0% | -0.5% | +5.5% |
| 2 to 4 waves | 68 | 20 | 58 | +0.0% | -0.3% | +0.9% |
| 4 to 8 waves | 67 | 17 | 63 | +0.0% | +0.0% | +0.5% |
| 8 waves or more | 90 | 6 | 89 | +0.0% | +0.0% | +0.0% |

### By auto's pick and capped's pick

| | shapes | capped faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| auto data-parallel, capped data-parallel | 272 | 0 | 272 | +0.0% | +0.0% | +0.0% |
| auto data-parallel, capped spread | 2 | 2 | 0 | +3.2% | +1.1% | +5.3% |
| auto one-tile owner, capped data-parallel | 45 | 45 | 0 | +27.4% | +5.6% | +69.5% |
| auto one-tile owner, capped one-tile owner | 227 | 160 | 145 | +0.3% | -0.5% | +8.9% |
| auto one-tile reduce-scatter, capped data-parallel | 1 | 1 | 0 | +79.9% | +79.9% | +79.9% |
| auto one-tile reduce-scatter, capped one-tile rs | 74 | 66 | 2 | +16.2% | -0.9% | +33.6% |
| auto two-tile (rs), capped one-tile rs | 9 | 4 | 3 | -0.2% | -2.7% | +1.0% |

## Wrong outputs, failures and notes

None: every v20 output matched the reference, v20 matched again after timing, and its flags and barrier counts were all back at 0.
