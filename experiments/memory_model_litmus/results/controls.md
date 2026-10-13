HIP_VISIBLE_DEVICES=1, AMD Instinct MI355X, triton 3.8.0, 2026-10-06 11:25

```text
GPU[0]		: GPU use (%): 0
GPU[1]		: GPU use (%): 0
GPU[2]		: GPU use (%): 0
GPU[3]		: GPU use (%): 89
GPU[4]		: GPU use (%): 0
GPU[5]		: GPU use (%): 0
GPU[6]		: GPU use (%): 0
GPU[7]		: GPU use (%): 0
================================== GPUs Indexed by PID ===================================
PID 2944433 is using 0 DRM device(s)
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

## controls

100 launches per row, a 1 GiB read before each. Fresh / stale / other are % of lines (64 per pair: the timed line 0 and 63 bulk lines).

| control | placement | expected | fresh % | stale % | other % | not fresh / lines | checks |
|---|---|---|---|---|---|---|---|
| negative: default store, default load | other XCD | stale | 0.00 | 100.00 | 0.00 | 819200 / 819200 | ok |
| negative: default store, default load | same XCD | stale (L1) | 0.00 | 100.00 | 0.00 | 819200 / 819200 | ok |
| two kernels: write in one launch, read in the next, default bits | other XCD | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | ok |
| two kernels, default bits | same XCD | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | ok |
| two kernels, default bits | same workgroup | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | ok |
| agent release + agent acquire (asm), default bits | other XCD | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | ok |
| agent release + agent acquire (compiler), default bits | other XCD | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | ok |
| agent release + agent acquire (asm), default bits | same XCD | fresh | 100.00 | 0.00 | 0.00 | 0 / 819200 | ok |
| agent release + agent acquire (asm), default bits | same workgroup | fresh | 100.00 | 0.00 | 0.00 | 0 / 1638400 | ok |
