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
PID 3172938 is using 1 DRM device(s):
5 
PID 2595882 is using 1 DRM device(s):
3 
PID 3172936 is using 1 DRM device(s):
4 
PID 2947934 is using 0 DRM device(s)
```

## stage2a

Fences with the time gap kept: the fence alone has to make the data visible. 50 launches per cell. Writer fence after the data stores; reader fence just before the read. `agent` is the inline-asm sequence LLVM emits (release `buffer_wbl2 sc1; s_waitcnt vmcnt(0)`, acquire `s_waitcnt vmcnt(0); buffer_inv sc1`); `agent_cc` is the compiler's own, from an atomic with `sem=release` / `acquire`, `scope=gpu`; `wg` is `s_waitcnt vmcnt(0)`; `inv_sc0` is `buffer_inv sc0`.


### none loads and stores, same workgroup

| writer fence | reader fence | observed | fresh % | stale % | not fresh / lines | checks |
|---|---|---|---|---|---|---|
| none | none | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| none | inv_sc0 | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| none | agent | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| none | agent_cc | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| wg | none | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| wg | inv_sc0 | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| wg | agent | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| wg | agent_cc | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| agent | none | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| agent | inv_sc0 | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| agent | agent | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| agent | agent_cc | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| agent_cc | none | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| agent_cc | inv_sc0 | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| agent_cc | agent | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| agent_cc | agent_cc | fresh | 100.00 | 0.00 | 0 / 819200 | ok |

### none loads and stores, same XCD

| writer fence | reader fence | observed | fresh % | stale % | not fresh / lines | checks |
|---|---|---|---|---|---|---|
| none | none | stale | 0.00 | 100.00 | 409600 / 409600 | ok |
| none | inv_sc0 | stale | 0.00 | 100.00 | 409600 / 409600 | ok |
| none | agent | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| none | agent_cc | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| wg | none | stale | 0.00 | 100.00 | 409600 / 409600 | ok |
| wg | inv_sc0 | stale | 0.00 | 100.00 | 409600 / 409600 | ok |
| wg | agent | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| wg | agent_cc | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| agent | none | stale | 0.00 | 100.00 | 409600 / 409600 | ok |
| agent | inv_sc0 | stale | 0.00 | 100.00 | 409600 / 409600 | ok |
| agent | agent | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| agent | agent_cc | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| agent_cc | none | stale | 0.00 | 100.00 | 409600 / 409600 | ok |
| agent_cc | inv_sc0 | stale | 0.00 | 100.00 | 409600 / 409600 | ok |
| agent_cc | agent | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| agent_cc | agent_cc | fresh | 100.00 | 0.00 | 0 / 409600 | ok |

### none loads and stores, other XCD

| writer fence | reader fence | observed | fresh % | stale % | not fresh / lines | checks |
|---|---|---|---|---|---|---|
| none | none | stale | 0.00 | 100.00 | 409600 / 409600 | ok |
| none | inv_sc0 | stale | 0.00 | 100.00 | 409600 / 409600 | ok |
| none | agent | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| none | agent_cc | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| wg | none | stale | 0.00 | 100.00 | 409600 / 409600 | ok |
| wg | inv_sc0 | stale | 0.00 | 100.00 | 409600 / 409600 | ok |
| wg | agent | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| wg | agent_cc | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| agent | none | stale | 0.00 | 100.00 | 409600 / 409600 | ok |
| agent | inv_sc0 | stale | 0.00 | 100.00 | 409600 / 409600 | ok |
| agent | agent | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| agent | agent_cc | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| agent_cc | none | stale | 0.00 | 100.00 | 409600 / 409600 | ok |
| agent_cc | inv_sc0 | stale | 0.00 | 100.00 | 409600 / 409600 | ok |
| agent_cc | agent | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| agent_cc | agent_cc | fresh | 100.00 | 0.00 | 0 / 409600 | ok |

### `sc0 nt` loads and stores, same workgroup

| writer fence | reader fence | observed | fresh % | stale % | not fresh / lines | checks |
|---|---|---|---|---|---|---|
| none | none | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| none | inv_sc0 | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| none | agent | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| none | agent_cc | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| wg | none | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| wg | inv_sc0 | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| wg | agent | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| wg | agent_cc | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| agent | none | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| agent | inv_sc0 | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| agent | agent | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| agent | agent_cc | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| agent_cc | none | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| agent_cc | inv_sc0 | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| agent_cc | agent | fresh | 100.00 | 0.00 | 0 / 819200 | ok |
| agent_cc | agent_cc | fresh | 100.00 | 0.00 | 0 / 819200 | ok |

### `sc0 nt` loads and stores, same XCD

| writer fence | reader fence | observed | fresh % | stale % | not fresh / lines | checks |
|---|---|---|---|---|---|---|
| none | none | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| none | inv_sc0 | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| none | agent | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| none | agent_cc | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| wg | none | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| wg | inv_sc0 | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| wg | agent | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| wg | agent_cc | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| agent | none | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| agent | inv_sc0 | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| agent | agent | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| agent | agent_cc | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| agent_cc | none | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| agent_cc | inv_sc0 | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| agent_cc | agent | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| agent_cc | agent_cc | fresh | 100.00 | 0.00 | 0 / 409600 | ok |

### `sc0 nt` loads and stores, other XCD

| writer fence | reader fence | observed | fresh % | stale % | not fresh / lines | checks |
|---|---|---|---|---|---|---|
| none | none | mixed | 29.07 | 70.93 | 290528 / 409600 | ok |
| none | inv_sc0 | mixed | 29.39 | 70.61 | 289239 / 409600 | ok |
| none | agent | mixed | 99.95 | 0.05 | 191 / 409600 | ok |
| none | agent_cc | mixed | 100.00 | 0.00 | 4 / 409600 | ok |
| wg | none | mixed | 28.16 | 71.84 | 294271 / 409600 | ok |
| wg | inv_sc0 | mixed | 28.93 | 71.07 | 291101 / 409600 | ok |
| wg | agent | mixed | 99.96 | 0.04 | 176 / 409600 | ok |
| wg | agent_cc | mixed | 99.99 | 0.01 | 34 / 409600 | ok |
| agent | none | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| agent | inv_sc0 | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| agent | agent | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| agent | agent_cc | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| agent_cc | none | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| agent_cc | inv_sc0 | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| agent_cc | agent | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
| agent_cc | agent_cc | fresh | 100.00 | 0.00 | 0 / 409600 | ok |
