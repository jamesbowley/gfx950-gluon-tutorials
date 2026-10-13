HIP_VISIBLE_DEVICES=3, AMD Instinct MI355X, triton 3.8.0, 2026-10-07 11:51

```text
GPU[0]		: GPU use (%): 0
GPU[1]		: GPU use (%): 0
GPU[2]		: GPU use (%): 0
GPU[3]		: GPU use (%): 0
GPU[4]		: GPU use (%): 0
GPU[5]		: GPU use (%): 0
GPU[6]		: GPU use (%): 0
GPU[7]		: GPU use (%): 0
================================== GPUs Indexed by PID ===================================
PID 1240294 is using 0 DRM device(s)
PID 1240283 is using 0 DRM device(s)
```

## primebits

`sc0 sc1` (`.wt`) stores, no fences, time-separated as in stage 1. The prime bits vary separately from the load bits; 'none' primes with default loads, 'no prime' skips the prime (phases 6: an old line then counts as other, not stale). 'writer reads too' (phases 15) has the writer read its own lines with the prime bits at prime time. 100 launches per cell, a 1 GiB read before each. Reference tiers (median ns): L1 42, L2 77, LLC 280, HBM 431.

| placement | prime | writer reads too | load | observed | fresh % | stale % | other % | not fresh / lines | timed read ns | tier | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| same workgroup | `sc0 nt` | no | `sc0 nt` | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | 425 | HBM | ok |
| same workgroup | none | no | `sc0 nt` | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | 279 | LLC | ok |
| same workgroup | `sc0 sc1` | no | `sc0 nt` | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | 279 | LLC | ok |
| same XCD | `sc0 nt` | no | `sc0 nt` | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 413 | HBM | ok |
| same XCD | none | no | `sc0 nt` | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 279 | LLC | ok |
| same XCD | `sc0 sc1` | no | `sc0 nt` | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 280 | LLC | ok |
| other XCD | `sc0 nt` | no | `sc0 nt` | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 288 | LLC | ok |
| other XCD | none | no | `sc0 nt` | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 282 | LLC | ok |
| other XCD | `sc0 sc1` | no | `sc0 nt` | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 283 | LLC | ok |
| same workgroup | no prime | no | `sc0 nt` | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | 429 | HBM | ok |
| same XCD | no prime | no | `sc0 nt` | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 430 | HBM | ok |
| other XCD | no prime | no | `sc0 nt` | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 436 | HBM | ok |
| same workgroup | no prime | no | `sc1` | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | 429 | HBM | ok |
| same XCD | no prime | no | `sc1` | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 429 | HBM | ok |
| other XCD | no prime | no | `sc1` | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 436 | HBM | ok |
| same workgroup | no prime | no | `sc0 sc1` | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | 429 | HBM | ok |
| same XCD | no prime | no | `sc0 sc1` | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 430 | HBM | ok |
| other XCD | no prime | no | `sc0 sc1` | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 435 | HBM | ok |
| same XCD | `sc0 nt` | no | `sc0 sc1` | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 413 | HBM | ok |
| same XCD | `sc0 nt` | no | `sc1` | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 414 | HBM | ok |
| other XCD | `sc0 nt` | yes | `sc0 nt` | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 320 | LLC | ok |
| other XCD | none | yes | `sc0 nt` | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 283 | LLC | ok |
