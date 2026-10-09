HIP_VISIBLE_DEVICES=1, AMD Instinct MI355X, triton 3.8.0, 2026-10-06 11:32

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
PID 3009033 is using 0 DRM device(s)
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

## stage2b

Ordering: no time gap. The writer stores, applies its fence, then sets flag[pair] = epoch with a relaxed `gpu`-scope atomic; the reader, which has been spinning since its prime, reads at once. The writer starts 20 us + 0-10 us of per-pair jitter after the grid barrier. Same workgroup: an `s_barrier` instead of the flag. 300 launches per row.

| placement | store | load | writer fence | reader fence | sync | what | observed | fresh % | not fresh / lines | median spins | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| other XCD | `sc0 sc1` | `sc0 sc1` | wg | none | flag | v21 `wt`: write-through, drain, relaxed flag, `.cv` reads | fresh | 100.000 | 0 / 2457600 | 120 | ok |
| other XCD | `sc0 sc1` | `sc0 sc1` | none | none | flag | the same without the drain before the flag | fresh | 100.000 | 0 / 2457600 | 119 | ok |
| other XCD | `sc1` | `sc1` | wg | none | flag | device scope only (`sc1`), drain, relaxed flag | fresh | 100.000 | 0 / 2457600 | 120 | ok |
| other XCD | `sc1` | `sc1` | none | none | flag | the same without the drain | fresh | 100.000 | 0 / 2457600 | 119 | ok |
| other XCD | none | none | agent | agent | flag | default bits, agent release and acquire fences around a relaxed flag | fresh | 100.000 | 0 / 2457600 | 136 | ok |
| other XCD | none | `sc0 nt` | agent | none | flag | default stores, agent release, flag, `sc0 nt` loads (no acquire) | fresh | 100.000 | 0 / 2457600 | 159 | ok |
| other XCD | `sc0 sc1` | `sc0 nt` | wg | none | flag | write-through stores, drain, flag, `sc0 nt` loads | fresh | 100.000 | 0 / 2457600 | 120 | ok |
| other XCD | none | none | none | agent | flag | default bits, acquire only (negative: no release) | stale | 0.000 | 2457600 / 2457600 | 118 | ok |
| other XCD | none | none | wg | none | flag | default bits, drain only (negative) | stale | 0.000 | 2457600 / 2457600 | 118 | ok |
| same XCD | none | `sc0 nt` | wg | none | flag | v21 `l2`: default stores, drain, flag, `sc0 nt` reads | fresh | 100.000 | 0 / 2457600 | 116 | ok |
| same XCD | none | `sc0 nt` | none | none | flag | the same without the drain | fresh | 100.000 | 0 / 2457600 | 109 | ok |
| same XCD | none | none | wg | none | flag | default loads (L1 can be stale) | mixed | 26.660 | 1802400 / 2457600 | 116 | ok |
| same XCD | none | none | wg | agent | flag | default loads after an agent acquire | fresh | 100.000 | 0 / 2457600 | 116 | ok |
| same workgroup | none | none | wg | none | barrier | same workgroup: drain, s_barrier, default loads | fresh | 100.000 | 0 / 4915200 | 0 | ok |
| same workgroup | none | none | none | none | barrier | the same without the drain | fresh | 100.000 | 0 / 4915200 | 0 | ok |
| same workgroup | none | `sc0 nt` | none | none | barrier | no drain, `sc0 nt` loads | fresh | 100.000 | 0 / 4915200 | 0 | ok |
| other XCD | `sc0 sc1` | `sc0 sc1` | wg | none | flag | v21 `wt`, stores queued ahead | fresh | 100.000 | 0 / 2457600 | 122 | ok |
| other XCD | `sc0 sc1` | `sc0 sc1` | none | none | flag | write-through, no drain, stores queued ahead | fresh | 100.000 | 0 / 2457600 | 122 | ok |
| other XCD | `sc1` | `sc1` | none | none | flag | `sc1`, no drain, stores queued ahead | fresh | 100.000 | 0 / 2457600 | 122 | ok |
| other XCD | none | none | agent | agent | flag | default bits, agent fences, stores queued ahead | fresh | 100.000 | 0 / 2457600 | 185 | ok |
| other XCD | none | `sc0 nt` | agent | none | flag | agent release, `sc0 nt` loads, stores queued ahead | fresh | 100.000 | 0 / 2457600 | 192 | ok |
| other XCD | `sc0 sc1` | `sc0 nt` | wg | none | flag | write-through, drain, `sc0 nt` loads, stores queued ahead | fresh | 100.000 | 0 / 2457600 | 123 | ok |
| same XCD | none | `sc0 nt` | wg | none | flag | v21 `l2`, stores queued ahead | fresh | 100.000 | 0 / 2457600 | 122 | ok |
| same XCD | none | `sc0 nt` | none | none | flag | v21 `l2` without the drain, stores queued ahead | fresh | 100.000 | 0 / 2457600 | 119 | ok |
| same workgroup | none | none | none | none | barrier | same workgroup, no drain, stores queued ahead | fresh | 100.000 | 0 / 4915200 | 0 | ok |
