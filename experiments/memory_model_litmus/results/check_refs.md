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
PID 3172935 is using 1 DRM device(s):
7 
PID 3172190 is using 0 DRM device(s)
PID 2595256 is using 1 DRM device(s):
3 
PID 2943879 is using 0 DRM device(s)
PID 3172938 is using 1 DRM device(s):
5 
PID 2595882 is using 1 DRM device(s):
3 
PID 3172936 is using 1 DRM device(s):
4 
```

## check

Placement: every program's XCC and CU from `HW_REG_XCC_ID` / `HW_REG_HW_ID`, 20 launches per placement, default bits.

| placement | pairs | writer XCDs | reader XCDs | pairs misplaced | launches with two workgroups on one CU |
|---|---|---|---|---|---|
| same workgroup | 256 | [0, 1, 2, 3, 4, 5, 6, 7] | [0, 1, 2, 3, 4, 5, 6, 7] | 0 | 0 |
| same XCD | 128 | [0, 1, 2, 3, 4, 5, 6, 7] | [0, 1, 2, 3, 4, 5, 6, 7] | 0 | 0 |
| other XCD | 128 | [0, 2, 4, 6] | [1, 3, 5, 7] | 0 | 0 |

Encodings: the cache bits on the experiment's own `global_load_dword` / `global_store_dword` (64-bit address, `off`) in each compiled stage-1 cell. Stores include the pollution store (no bits).

| store | load | load bits in the asm | store bits in the asm | as named |
|---|---|---|---|---|
| none | none | [''] | [''] | True |
| none | `sc0 nt` | ['sc0 nt'] | [''] | True |
| none | `sc1` | ['sc1'] | [''] | True |
| none | `sc0 sc1` | ['sc0 sc1'] | [''] | True |
| `sc0 nt` | none | [''] | ['', 'sc0 nt'] | True |
| `sc0 nt` | `sc0 nt` | ['sc0 nt'] | ['', 'sc0 nt'] | True |
| `sc0 nt` | `sc1` | ['sc1'] | ['', 'sc0 nt'] | True |
| `sc0 nt` | `sc0 sc1` | ['sc0 sc1'] | ['', 'sc0 nt'] | True |
| `sc1` | none | [''] | ['', 'sc1'] | True |
| `sc1` | `sc0 nt` | ['sc0 nt'] | ['', 'sc1'] | True |
| `sc1` | `sc1` | ['sc1'] | ['', 'sc1'] | True |
| `sc1` | `sc0 sc1` | ['sc0 sc1'] | ['', 'sc1'] | True |
| `sc0 sc1` | none | [''] | ['', 'sc0 sc1'] | True |
| `sc0 sc1` | `sc0 nt` | ['sc0 nt'] | ['', 'sc0 sc1'] | True |
| `sc0 sc1` | `sc1` | ['sc1'] | ['', 'sc0 sc1'] | True |
| `sc0 sc1` | `sc0 sc1` | ['sc0 sc1'] | ['', 'sc0 sc1'] | True |

## refs

The timed read of line 0 (one wave-uniform line), with nothing written. Bracket-subtracted `s_memtime` cycles converted to ns with each wave's own clock (s_memtime against s_memrealtime over the kernel); medians and 5th-95th percentiles over 50 launches x 128 pairs. `rt` is the same read timed with `s_memrealtime` (10 ns ticks, includes the s_memtime round trips).

| case | ns | p5-p95 ns | cycles | rt ns | clock MHz |
|---|---|---|---|---|---|
| L1: the reader read the line (default), default read | 42 | 38-44 | 100 | 80 | 2395 |
| L2: another CU of the XCD read the line, default read | 77 | 69-87 | 184 | 80 | 2393 |
| LLC: another XCD read the line, default read | 283 | 228-357 | 676 | 320 | 2385 |
| HBM: 1 GiB read first, default read | 433 | 335-506 | 1036 | 440 | 2390 |
| L2: the reader read the line (default), `sc0 nt` read | 77 | 69-87 | 184 | 80 | 2394 |
| L2: the reader read the line (default), `sc1` read | 78 | 68-87 | 188 | 80 | 2394 |
| L2: the reader read the line (default), `sc0 sc1` read | 77 | 69-87 | 184 | 80 | 2394 |
| L2: another CU of the XCD read the line, `sc1` read | 77 | 68-87 | 184 | 80 | 2396 |
| LLC: another XCD read the line, `sc1` read | 286 | 228-356 | 684 | 320 | 2390 |
| LLC: another XCD read the line, `sc0 sc1` read | 285 | 228-356 | 680 | 320 | 2389 |
| HBM: 1 GiB read first, `sc1` read | 433 | 335-505 | 1036 | 440 | 2393 |
| HBM: 1 GiB read first, `sc0 sc1` read | 433 | 332-511 | 1036 | 440 | 2395 |
| (no flush) the reader read the line with `sc0 sc1`, `sc0 sc1` read | 77 | 69-87 | 184 | 80 | 2382 |
| (no flush) the reader read the line with `sc1`, `sc1` read | 77 | 69-87 | 184 | 80 | 2381 |

Empty bracket (268800 samples): median 44 cycles, p5 44, p25 44, p75 44, p95 44, max 52. At the median clock (2389 MHz) the p5-p95 spread is 0 ns.
Smallest gap between tier medians: 36 ns.
