# v20 stream-K policies on the 630 shapes (v20 only)

Shapes: `supported_status_scoped.csv`, timed as in the AITER comparison (`do_bench_cudagraph`, 2 interleaved rounds, median) but without the AITER worker (`sweep --v20-only`).

- Stack: torch 2.13.0+rocm7.14.0, Triton 3.8.0 @ /home/jbowley/repos/triton_gfx950-tutorial-v2.2/python/triton.
- GPU: HIP device 0, PCI bus 117, gfx950:sramecc+:xnack-.
- v20: `v20_streamk_reduce_scatter` (snapshot sha256 5f8dd0a50013c816), v9 tile order.

Measured 630 shapes.

## Diagnostic: auto's stream-K choice against forcing data-parallel

On the 356 shapes where auto picks a stream-K path. Positive means auto's choice is faster than v20 `dp`; this column never counts as v20's result.

| | shapes | auto faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| all | 356 | 289 | 8 | +19.1% | -15.4% | +104.0% |
| one-tile owner, K < 2048 | 42 | 0 | 0 | -24.2% | -43.7% | -8.9% |
| one-tile owner, K >= 2048 | 230 | 213 | 8 | +16.9% | +1.9% | +70.0% |
| one-tile reduce-scatter, K < 2048 | 7 | 0 | 0 | -24.3% | -24.8% | -21.5% |
| one-tile reduce-scatter, K >= 2048 | 68 | 67 | 0 | +99.6% | +44.0% | +150.3% |
| two-tile (rs), K >= 2048 | 9 | 9 | 0 | +12.5% | +6.1% | +27.1% |

## v20 capped against v20 auto

`STREAMK_POLICY=capped` against the default rule. Where capped picks data-parallel its time is the data-parallel column (the same kernel). Positive means capped is faster.

| | shapes | capped faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| all shapes | 630 | 308 | 417 | +0.0% | +0.0% | +23.3% |
| under 1 wave | 321 | 234 | 144 | +1.9% | +0.0% | +37.1% |
| 1 to 2 waves | 84 | 28 | 63 | +0.0% | -0.1% | +6.0% |
| 2 to 4 waves | 68 | 24 | 58 | +0.0% | -0.1% | +0.9% |
| 4 to 8 waves | 67 | 14 | 63 | +0.0% | +0.0% | +0.3% |
| 8 waves or more | 90 | 8 | 89 | +0.0% | +0.0% | +0.0% |

### By auto's pick and capped's pick

| | shapes | capped faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| auto data-parallel, capped data-parallel | 272 | 0 | 272 | +0.0% | +0.0% | +0.0% |
| auto data-parallel, capped spread | 2 | 2 | 1 | +2.5% | +0.8% | +4.2% |
| auto one-tile owner, capped data-parallel | 51 | 49 | 1 | +22.8% | +3.8% | +74.9% |
| auto one-tile owner, capped one-tile owner | 221 | 180 | 140 | +0.4% | -0.2% | +10.8% |
| auto one-tile reduce-scatter, capped data-parallel | 1 | 1 | 0 | +70.3% | +70.3% | +70.3% |
| auto one-tile reduce-scatter, capped one-tile rs | 74 | 71 | 2 | +21.5% | +1.8% | +45.8% |
| auto two-tile (rs), capped one-tile rs | 9 | 5 | 1 | +0.8% | -1.6% | +1.7% |

## Wrong outputs, failures and notes

None: every v20 output matched the reference, v20 matched again after timing, and its flags and barrier counts were all back at 0.
