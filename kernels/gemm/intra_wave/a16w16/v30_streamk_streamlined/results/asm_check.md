# v30 reduce-scatter synchronization, 256x256x64

| exchange | VGPRs | spills | atomics | buffer_wbl2 | buffer_inv | loads sc0 nt | loads sc0 sc1 | stores sc0 sc1 |
|---|---|---|---|---|---|---|---|---|
| wt | 488 | 0 | 5 | 0 | 0 | 0 | 64 | 64 |
| l2 | 488 | 0 | 0 | 0 | 0 | 17 | 0 | 0 |

l2: no atomics, no cache maintenance, sc0 nt loads present
# v30 reduce-scatter synchronization, 128x128x128

| exchange | VGPRs | spills | atomics | buffer_wbl2 | buffer_inv | loads sc0 nt | loads sc0 sc1 | stores sc0 sc1 |
|---|---|---|---|---|---|---|---|---|
| wt | 256 | 0 | 5 | 0 | 0 | 0 | 16 | 16 |
| l2 | 256 | 0 | 0 | 0 | 0 | 17 | 0 | 0 |

l2: no atomics, no cache maintenance, sc0 nt loads present
# v30 reduce-scatter synchronization, 64x64x64

| exchange | VGPRs | spills | atomics | buffer_wbl2 | buffer_inv | loads sc0 nt | loads sc0 sc1 | stores sc0 sc1 |
|---|---|---|---|---|---|---|---|---|
| wt | 84 | 0 | 5 | 0 | 0 | 0 | 8 | 4 |
| l2 | 84 | 0 | 0 | 0 | 0 | 9 | 0 | 0 |

l2: no atomics, no cache maintenance, sc0 nt loads present
