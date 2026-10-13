HIP_VISIBLE_DEVICES=1, AMD Instinct MI355X, triton 3.8.0, 2026-10-06 11:25

```text
GPU[0]		: GPU use (%): 0
GPU[1]		: GPU use (%): 0
GPU[2]		: GPU use (%): 0
GPU[3]		: GPU use (%): 96
GPU[4]		: GPU use (%): 0
GPU[5]		: GPU use (%): 0
GPU[6]		: GPU use (%): 0
GPU[7]		: GPU use (%): 0
================================== GPUs Indexed by PID ===================================
PID 2946238 is using 0 DRM device(s)
PID 3165867 is using 0 DRM device(s)
PID 3172937 is using 1 DRM device(s):
6 
PID 3172935 is using 1 DRM device(s):
7 
PID 3172190 is using 0 DRM device(s)
PID 2595256 is using 1 DRM device(s):
3 
PID 3172938 is using 1 DRM device(s):
5 
PID 2595882 is using 1 DRM device(s):
3 
PID 3172936 is using 1 DRM device(s):
4 
```

## stage1

No fences. Time-separated: prime at ts, write at ts + 20 us, read at ts + 70 us. 100 launches per cell, a 1 GiB read before each. Reference tiers (median ns): L1 42, L2 77, LLC 280, HBM 432.


### same workgroup

| store | load | predicted | observed | fresh % | stale % | other % | not fresh / lines | timed read ns | tier | timed: fresh ns | timed: stale ns | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | none | ? | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | 77 | L2 | 77 | nan | ok |
| none | `sc0 nt` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | 339 | LLC | 339 | nan | ok |
| none | `sc1` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | 78 | L2 | 78 | nan | ok |
| none | `sc0 sc1` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | 78 | L2 | 78 | nan | ok |
| `sc0 nt` | none | ? | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | 77 | L2 | 77 | nan | ok |
| `sc0 nt` | `sc0 nt` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | 335 | LLC | 335 | nan | ok |
| `sc0 nt` | `sc1` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | 78 | L2 | 78 | nan | ok |
| `sc0 nt` | `sc0 sc1` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | 78 | L2 | 78 | nan | ok |
| `sc1` | none | ? | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | 276 | LLC | 276 | nan | ok |
| `sc1` | `sc0 nt` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | 426 | HBM | 426 | nan | ok |
| `sc1` | `sc1` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | 277 | LLC | 277 | nan | ok |
| `sc1` | `sc0 sc1` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | 278 | LLC | 278 | nan | ok |
| `sc0 sc1` | none | ? | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | 275 | LLC | 275 | nan | ok |
| `sc0 sc1` | `sc0 nt` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | 426 | HBM | 426 | nan | ok |
| `sc0 sc1` | `sc1` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | 277 | LLC | 277 | nan | ok |
| `sc0 sc1` | `sc0 sc1` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | 277 | LLC | 277 | nan | ok |

### same XCD

| store | load | predicted | observed | fresh % | stale % | other % | not fresh / lines | timed read ns | tier | timed: fresh ns | timed: stale ns | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | none | stale | stale | 0.00 | 100.00 | 0.00 | 819200 / 819200 | 42 | L1 | nan | 42 | ok |
| none | `sc0 nt` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 78 | L2 | 78 | nan | ok |
| none | `sc1` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 77 | L2 | 77 | nan | ok |
| none | `sc0 sc1` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 78 | L2 | 78 | nan | ok |
| `sc0 nt` | none | stale | stale | 0.00 | 100.00 | 0.00 | 819200 / 819200 | 42 | L1 | nan | 42 | ok |
| `sc0 nt` | `sc0 nt` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 77 | L2 | 77 | nan | ok |
| `sc0 nt` | `sc1` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 78 | L2 | 78 | nan | ok |
| `sc0 nt` | `sc0 sc1` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 78 | L2 | 78 | nan | ok |
| `sc1` | none | stale | stale | 0.00 | 100.00 | 0.00 | 819200 / 819200 | 42 | L1 | nan | 42 | ok |
| `sc1` | `sc0 nt` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 413 | HBM | 413 | nan | ok |
| `sc1` | `sc1` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 280 | LLC | 280 | nan | ok |
| `sc1` | `sc0 sc1` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 280 | LLC | 280 | nan | ok |
| `sc0 sc1` | none | stale | stale | 0.00 | 100.00 | 0.00 | 819200 / 819200 | 42 | L1 | nan | 42 | ok |
| `sc0 sc1` | `sc0 nt` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 414 | HBM | 414 | nan | ok |
| `sc0 sc1` | `sc1` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 279 | LLC | 279 | nan | ok |
| `sc0 sc1` | `sc0 sc1` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 280 | LLC | 280 | nan | ok |

### other XCD

| store | load | predicted | observed | fresh % | stale % | other % | not fresh / lines | timed read ns | tier | timed: fresh ns | timed: stale ns | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | none | stale | stale | 0.00 | 100.00 | 0.00 | 819200 / 819200 | 42 | L1 | nan | 42 | ok |
| none | `sc0 nt` | stale | stale | 0.00 | 100.00 | 0.00 | 819200 / 819200 | 78 | L2 | nan | 78 | ok |
| none | `sc1` | stale | stale | 0.00 | 100.00 | 0.00 | 819200 / 819200 | 78 | L2 | nan | 78 | ok |
| none | `sc0 sc1` | stale | stale | 0.00 | 100.00 | 0.00 | 819200 / 819200 | 78 | L2 | nan | 78 | ok |
| `sc0 nt` | none | stale | stale | 0.00 | 100.00 | 0.00 | 819200 / 819200 | 42 | L1 | nan | 42 | ok |
| `sc0 nt` | `sc0 nt` | stale | mixed | 30.45 | 69.55 | 0.00 | 569716 / 819200 | 82 | L2 | 427 | 79 | ok |
| `sc0 nt` | `sc1` | stale | mixed | 32.06 | 67.94 | 0.00 | 556568 / 819200 | 82 | L2 | 431 | 79 | ok |
| `sc0 nt` | `sc0 sc1` | stale | mixed | 32.96 | 67.04 | 0.00 | 549181 / 819200 | 82 | L2 | 428 | 79 | ok |
| `sc1` | none | stale | stale | 0.00 | 100.00 | 0.00 | 819200 / 819200 | 42 | L1 | nan | 42 | ok |
| `sc1` | `sc0 nt` | stale | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 286 | LLC | 286 | nan | ok |
| `sc1` | `sc1` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 282 | LLC | 282 | nan | ok |
| `sc1` | `sc0 sc1` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 282 | LLC | 282 | nan | ok |
| `sc0 sc1` | none | stale | stale | 0.00 | 100.00 | 0.00 | 819200 / 819200 | 42 | L1 | nan | 42 | ok |
| `sc0 sc1` | `sc0 nt` | stale | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 289 | LLC | 289 | nan | ok |
| `sc0 sc1` | `sc1` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 282 | LLC | 282 | nan | ok |
| `sc0 sc1` | `sc0 sc1` | fresh | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | 282 | LLC | 282 | nan | ok |
