# v30 reduce-scatter synchronization with rs_all (WORKLOG entry 18)

asm_check.py, one-tile rs split 8 on 17x16 tiles, K = 8192. wt atomics double (5 to 10) where idle programs read: their path has its own copy of the tile barrier. 64x64x64 has E = 0 here (4 blocks).

## v30 reduce-scatter synchronization, 256x256x64

| exchange | VGPRs | spills | atomics | buffer_wbl2 | buffer_inv | loads sc0 nt | loads sc0 sc1 | stores sc0 sc1 |
|---|---|---|---|---|---|---|---|---|
| wt | 484 | 0 | 10 | 0 | 0 | 0 | 64 | 64 |
| l2 | 480 | 0 | 0 | 0 | 0 | 17 | 0 | 0 |

l2: no atomics, no cache maintenance, sc0 nt loads present

## v30 reduce-scatter synchronization, 256x128x64

| exchange | VGPRs | spills | atomics | buffer_wbl2 | buffer_inv | loads sc0 nt | loads sc0 sc1 | stores sc0 sc1 |
|---|---|---|---|---|---|---|---|---|
| wt | 292 | 0 | 10 | 0 | 0 | 0 | 32 | 32 |
| l2 | 292 | 0 | 0 | 0 | 0 | 17 | 0 | 0 |

l2: no atomics, no cache maintenance, sc0 nt loads present

## v30 reduce-scatter synchronization, 128x256x64

| exchange | VGPRs | spills | atomics | buffer_wbl2 | buffer_inv | loads sc0 nt | loads sc0 sc1 | stores sc0 sc1 |
|---|---|---|---|---|---|---|---|---|
| wt | 372 | 0 | 10 | 0 | 0 | 0 | 32 | 32 |
| l2 | 368 | 0 | 0 | 0 | 0 | 17 | 0 | 0 |

l2: no atomics, no cache maintenance, sc0 nt loads present

## v30 reduce-scatter synchronization, 128x128x128

| exchange | VGPRs | spills | atomics | buffer_wbl2 | buffer_inv | loads sc0 nt | loads sc0 sc1 | stores sc0 sc1 |
|---|---|---|---|---|---|---|---|---|
| wt | 260 | 0 | 10 | 0 | 0 | 0 | 16 | 16 |
| l2 | 260 | 0 | 0 | 0 | 0 | 17 | 0 | 0 |

l2: no atomics, no cache maintenance, sc0 nt loads present

## v30 reduce-scatter synchronization, 64x64x64

| exchange | VGPRs | spills | atomics | buffer_wbl2 | buffer_inv | loads sc0 nt | loads sc0 sc1 | stores sc0 sc1 |
|---|---|---|---|---|---|---|---|---|
| wt | 80 | 0 | 5 | 0 | 0 | 0 | 8 | 4 |
| l2 | 80 | 0 | 0 | 0 | 0 | 9 | 0 | 0 |

l2: no atomics, no cache maintenance, sc0 nt loads present
