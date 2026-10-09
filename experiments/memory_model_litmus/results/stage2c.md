HIP_VISIBLE_DEVICES=1, AMD Instinct MI355X, triton 3.8.0, 2026-10-06 11:24

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
PID 2941239 is using 0 DRM device(s)
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

## stage2c

The flag atomic's own scope and order. First the code each (scope, sem) produces; then runs.

| scope | writer sem | reader sem | distinct instructions of the flag path (sync-related lines in the whole kernel) |
|---|---|---|---|
| cta | relaxed | relaxed | `global_atomic_add v3, v4, s[12:13]`; `global_atomic_swap v2, v3, s[56:57]`; `global_load_dword v2, v2, s[8:9] sc0`; `global_load_dword v4, v2, s[12:13] sc1`; `global_load_dword v6, v3, s[8:9] sc0`; `s_waitcnt vmcnt(0)`; `s_waitcnt vmcnt(0) lgkmcnt(0)` |
| cta | release | acquire | `global_atomic_add v3, v4, s[12:13]`; `global_atomic_swap v2, v3, s[56:57]`; `global_load_dword v2, v2, s[8:9] sc0`; `global_load_dword v4, v2, s[12:13] sc1`; `global_load_dword v6, v3, s[8:9] sc0`; `s_waitcnt vmcnt(0)`; `s_waitcnt vmcnt(0) lgkmcnt(0)` |
| gpu | relaxed | relaxed | `global_atomic_add v3, v4, s[12:13]`; `global_atomic_swap v2, v3, s[56:57]`; `global_load_dword v2, v2, s[8:9] sc1`; `global_load_dword v4, v2, s[12:13] sc1`; `global_load_dword v6, v3, s[8:9] sc1`; `s_waitcnt vmcnt(0)`; `s_waitcnt vmcnt(0) lgkmcnt(0)` |
| gpu | release | acquire | `buffer_inv sc1`; `buffer_wbl2 sc1`; `global_atomic_add v3, v4, s[12:13]`; `global_atomic_swap v2, v3, s[56:57]`; `global_load_dword v2, v2, s[8:9] sc1`; `global_load_dword v4, v2, s[12:13] sc1`; `global_load_dword v6, v3, s[8:9] sc1`; `s_waitcnt vmcnt(0)`; `s_waitcnt vmcnt(0) lgkmcnt(0)` |
| sys | relaxed | relaxed | `global_atomic_add v3, v4, s[12:13]`; `global_atomic_swap v2, v3, s[56:57] sc1`; `global_load_dword v2, v2, s[8:9] sc0 sc1`; `global_load_dword v4, v2, s[12:13] sc1`; `global_load_dword v6, v3, s[8:9] sc0 sc1`; `s_waitcnt vmcnt(0)`; `s_waitcnt vmcnt(0) lgkmcnt(0)` |
| sys | release | acquire | `buffer_inv sc0 sc1`; `buffer_wbl2 sc0 sc1`; `global_atomic_add v3, v4, s[12:13]`; `global_atomic_swap v2, v3, s[56:57] sc1`; `global_load_dword v2, v2, s[8:9] sc0 sc1`; `global_load_dword v4, v2, s[12:13] sc1`; `global_load_dword v6, v3, s[8:9] sc0 sc1`; `s_waitcnt vmcnt(0)`; `s_waitcnt vmcnt(0) lgkmcnt(0)` |

Runs, 100 launches per row, 0-10 us jitter. Data with write-through bits and a drain (visibility of the data is not in question, only the flag's), then default data bits, where the atomic's own release / acquire must also make the data visible.

| placement | data bits | scope | writer sem | reader sem | observed | fresh % | not fresh / lines | timeouts | median spins | checks |
|---|---|---|---|---|---|---|---|---|---|---|
| same XCD | `sc0 sc1` | cta | relaxed | relaxed | no data | nan | 0 / 0 | 12800 | 0 | 12800 timeouts |
| same XCD | `sc0 sc1` | cta | release | acquire | no data | nan | 0 / 0 | 12800 | 0 | 12800 timeouts |
| same XCD | `sc0 sc1` | gpu | relaxed | relaxed | fresh | 100.000 | 0 / 819200 | 0 | 117 | ok |
| same XCD | `sc0 sc1` | gpu | release | acquire | fresh | 100.000 | 0 / 819200 | 0 | 112 | ok |
| same XCD | `sc0 sc1` | sys | relaxed | relaxed | fresh | 100.000 | 0 / 819200 | 0 | 117 | ok |
| same XCD | `sc0 sc1` | sys | release | acquire | fresh | 100.000 | 0 / 819200 | 0 | 112 | ok |
| same XCD | none | cta | relaxed | relaxed | no data | nan | 0 / 0 | 12800 | 0 | 12800 timeouts |
| same XCD | none | cta | release | acquire | no data | nan | 0 / 0 | 12800 | 0 | 12800 timeouts |
| same XCD | none | gpu | relaxed | relaxed | mixed | 26.660 | 600800 / 819200 | 0 | 110 | ok |
| same XCD | none | gpu | release | acquire | fresh | 100.000 | 0 / 819200 | 0 | 113 | ok |
| same XCD | none | sys | relaxed | relaxed | mixed | 26.660 | 600800 / 819200 | 0 | 110 | ok |
| same XCD | none | sys | release | acquire | fresh | 100.000 | 0 / 819200 | 0 | 113 | ok |
| other XCD | `sc0 sc1` | cta | relaxed | relaxed | no data | nan | 0 / 0 | 12800 | 0 | 12800 timeouts |
| other XCD | `sc0 sc1` | cta | release | acquire | no data | nan | 0 / 0 | 12800 | 0 | 12800 timeouts |
| other XCD | `sc0 sc1` | gpu | relaxed | relaxed | fresh | 100.000 | 0 / 819200 | 0 | 120 | ok |
| other XCD | `sc0 sc1` | gpu | release | acquire | fresh | 100.000 | 0 / 819200 | 0 | 124 | ok |
| other XCD | `sc0 sc1` | sys | relaxed | relaxed | fresh | 100.000 | 0 / 819200 | 0 | 120 | ok |
| other XCD | `sc0 sc1` | sys | release | acquire | fresh | 100.000 | 0 / 819200 | 0 | 125 | ok |
| other XCD | none | cta | relaxed | relaxed | no data | nan | 0 / 0 | 12800 | 0 | 12800 timeouts |
| other XCD | none | cta | release | acquire | no data | nan | 0 / 0 | 12800 | 0 | 12800 timeouts |
| other XCD | none | gpu | relaxed | relaxed | stale | 0.000 | 819200 / 819200 | 0 | 119 | ok |
| other XCD | none | gpu | release | acquire | fresh | 100.000 | 0 / 819200 | 0 | 128 | ok |
| other XCD | none | sys | relaxed | relaxed | stale | 0.000 | 819200 / 819200 | 0 | 119 | ok |
| other XCD | none | sys | release | acquire | fresh | 100.000 | 0 / 819200 | 0 | 129 | ok |
