HIP_VISIBLE_DEVICES=1, AMD Instinct MI355X, triton 3.8.0, 2026-10-06 11:25

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
PID 2947694 is using 0 DRM device(s)
PID 3172938 is using 1 DRM device(s):
5 
PID 2595882 is using 1 DRM device(s):
3 
PID 3172936 is using 1 DRM device(s):
4 
```

## pollute

Writer programs store E MiB per writer XCD of other lines (default bits) right after the data. 50 launches per cell.

| placement | store | load | E MiB per writer XCD | fresh % | stale % | not fresh / lines | checks |
|---|---|---|---|---|---|---|---|
| other XCD | none | `sc1` | 0 | 0.00 | 100.00 | 409600 / 409600 | ok |
| other XCD | none | `sc1` | 1 | 0.33 | 99.67 | 408245 / 409600 | ok |
| other XCD | none | `sc1` | 2 | 50.00 | 50.00 | 204800 / 409600 | ok |
| other XCD | none | `sc1` | 4 | 100.00 | 0.00 | 0 / 409600 | ok |
| other XCD | none | `sc1` | 8 | 100.00 | 0.00 | 0 / 409600 | ok |
| other XCD | `sc0 nt` | `sc1` | 0 | 31.16 | 68.84 | 281988 / 409600 | ok |
| other XCD | `sc0 nt` | `sc1` | 1 | 99.61 | 0.39 | 1582 / 409600 | ok |
| other XCD | `sc0 nt` | `sc1` | 2 | 68.01 | 31.99 | 131025 / 409600 | ok |
| other XCD | `sc0 nt` | `sc1` | 4 | 100.00 | 0.00 | 0 / 409600 | ok |
| other XCD | `sc0 nt` | `sc1` | 8 | 100.00 | 0.00 | 0 / 409600 | ok |
| other XCD | none | `sc0 sc1` | 0 | 0.00 | 100.00 | 409600 / 409600 | ok |
| other XCD | none | `sc0 sc1` | 1 | 0.34 | 99.66 | 408219 / 409600 | ok |
| other XCD | none | `sc0 sc1` | 2 | 50.00 | 50.00 | 204800 / 409600 | ok |
| other XCD | none | `sc0 sc1` | 4 | 100.00 | 0.00 | 0 / 409600 | ok |
| other XCD | none | `sc0 sc1` | 8 | 100.00 | 0.00 | 0 / 409600 | ok |
| same XCD | none | none | 0 | 0.00 | 100.00 | 409600 / 409600 | ok |
| same XCD | none | none | 1 | 0.00 | 100.00 | 409600 / 409600 | ok |
| same XCD | none | none | 2 | 0.00 | 100.00 | 409600 / 409600 | ok |
| same XCD | none | none | 4 | 0.00 | 100.00 | 409600 / 409600 | ok |
| same XCD | none | none | 8 | 0.00 | 100.00 | 409600 / 409600 | ok |
