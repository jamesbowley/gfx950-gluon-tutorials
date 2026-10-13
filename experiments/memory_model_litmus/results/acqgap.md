HIP_VISIBLE_DEVICES=1, AMD Instinct MI355X, triton 3.8.0, 2026-10-06 11:31

```text
GPU[0]		: GPU use (%): 0
GPU[1]		: GPU use (%): 0
GPU[2]		: GPU use (%): 0
GPU[3]		: GPU use (%): 100
GPU[4]		: GPU use (%): 0
GPU[5]		: GPU use (%): 0
GPU[6]		: GPU use (%): 0
GPU[7]		: GPU use (%): 0
================================== GPUs Indexed by PID ===================================
PID 3165867 is using 0 DRM device(s)
PID 3172937 is using 1 DRM device(s):
6 
PID 3172935 is using 1 DRM device(s):
7 
PID 3172190 is using 0 DRM device(s)
PID 2595256 is using 1 DRM device(s):
3 
PID 3000545 is using 0 DRM device(s)
PID 3172938 is using 1 DRM device(s):
5 
PID 2595882 is using 1 DRM device(s):
3 
PID 3172936 is using 1 DRM device(s):
4 
```

## acqgap

The reader applies `buffer_inv sc1` + `s_waitcnt vmcnt(0)` gap us before its read at ts + 70 us (gap 0: just before). No writer fence. 30 launches per cell.

| placement | store | load | gap us | observed | fresh % | not fresh / lines | timed ns | tier |
|---|---|---|---|---|---|---|---|---|
| other XCD | `sc0 sc1` | none | 0 | fresh | 100.00 | 0 / 245760 | 1801 | HBM |
| other XCD | `sc0 sc1` | none | 0.5 | fresh | 100.00 | 0 / 245760 | 1767 | HBM |
| other XCD | `sc0 sc1` | none | 1 | fresh | 100.00 | 0 / 245760 | 1773 | HBM |
| other XCD | `sc0 sc1` | none | 2 | fresh | 100.00 | 0 / 245760 | 1697 | HBM |
| other XCD | `sc0 sc1` | none | 5 | fresh | 100.00 | 0 / 245760 | 283 | LLC |
| other XCD | `sc0 sc1` | none | 10 | fresh | 100.00 | 0 / 245760 | 282 | LLC |
| other XCD | `sc0 sc1` | none | 30 | fresh | 100.00 | 0 / 245760 | 282 | LLC |
| same XCD | none | none | 0 | fresh | 100.00 | 0 / 245760 | 783 | HBM |
| same XCD | none | none | 0.5 | fresh | 100.00 | 0 / 245760 | 688 | HBM |
| same XCD | none | none | 1 | fresh | 100.00 | 0 / 245760 | 691 | HBM |
| same XCD | none | none | 2 | fresh | 100.00 | 0 / 245760 | 78 | L2 |
| same XCD | none | none | 5 | fresh | 100.00 | 0 / 245760 | 78 | L2 |
| same XCD | none | none | 10 | fresh | 100.00 | 0 / 245760 | 78 | L2 |
| same XCD | none | none | 30 | fresh | 100.00 | 0 / 245760 | 78 | L2 |
| other XCD | none | none | 0 | fresh | 100.00 | 0 / 245760 | 1745 | HBM |
| other XCD | none | none | 0.5 | mixed | 100.00 | 8 / 245760 | 1726 | HBM |
| other XCD | none | none | 1 | mixed | 99.99 | 19 / 245760 | 1744 | HBM |
| other XCD | none | none | 2 | mixed | 98.28 | 4239 / 245760 | 1559 | HBM |
| other XCD | none | none | 5 | stale | 0.00 | 245760 / 245760 | 79 | L2 |
| other XCD | none | none | 10 | stale | 0.00 | 245760 / 245760 | 79 | L2 |
| other XCD | none | none | 30 | stale | 0.00 | 245760 / 245760 | 79 | L2 |
| other XCD | none | `sc0 nt` | 0 | mixed | 99.90 | 255 / 245760 | 1708 | HBM |
| other XCD | none | `sc0 nt` | 0.5 | mixed | 99.74 | 636 / 245760 | 1637 | HBM |
| other XCD | none | `sc0 nt` | 1 | mixed | 99.03 | 2387 / 245760 | 1667 | HBM |
| other XCD | none | `sc0 nt` | 2 | mixed | 3.78 | 236462 / 245760 | 1508 | HBM |
| other XCD | none | `sc0 nt` | 5 | stale | 0.00 | 245760 / 245760 | 79 | L2 |
| other XCD | none | `sc0 nt` | 10 | stale | 0.00 | 245760 / 245760 | 79 | L2 |
| other XCD | none | `sc0 nt` | 30 | stale | 0.00 | 245760 / 245760 | 79 | L2 |
