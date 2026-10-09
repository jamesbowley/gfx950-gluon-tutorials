HIP_VISIBLE_DEVICES=1, AMD Instinct MI355X, triton 3.8.0, 2026-10-06 11:26

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
PID 2950537 is using 0 DRM device(s)
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

## third

Default stores on XCDs 0/2/4/6, `sc0 nt` reads (primed) on XCDs 1/3/5/7, no fences. At ts + 40 us every program on one XCD issues the operation (then `s_waitcnt vmcnt(0)`). 50 launches.

| operation | issued on XCD | writer -> reader XCDs | relation of the issuing XCD | fresh % | not fresh / lines | op cycles (median) |
|---|---|---|---|---|---|---|
| `buffer_inv sc1` | 3 | 0 -> 1 | neither | 0.00 | 102400 / 102400 | 4598 |
| `buffer_inv sc1` | 3 | 2 -> 3 | the reader's | 0.00 | 102400 / 102400 | 4598 |
| `buffer_inv sc1` | 3 | 4 -> 5 | neither | 0.00 | 102400 / 102400 | 4598 |
| `buffer_inv sc1` | 3 | 6 -> 7 | neither | 0.00 | 102400 / 102400 | 4598 |
| `buffer_inv sc1` | 2 | 0 -> 1 | neither | 0.00 | 102400 / 102400 | 4660 |
| `buffer_inv sc1` | 2 | 2 -> 3 | the writer's | 0.00 | 102400 / 102400 | 4660 |
| `buffer_inv sc1` | 2 | 4 -> 5 | neither | 0.00 | 102400 / 102400 | 4660 |
| `buffer_inv sc1` | 2 | 6 -> 7 | neither | 0.00 | 102400 / 102400 | 4660 |
| `buffer_wbl2 sc1` | 3 | 0 -> 1 | neither | 0.00 | 102400 / 102400 | 4588 |
| `buffer_wbl2 sc1` | 3 | 2 -> 3 | the reader's | 0.00 | 102400 / 102400 | 4588 |
| `buffer_wbl2 sc1` | 3 | 4 -> 5 | neither | 0.00 | 102400 / 102400 | 4588 |
| `buffer_wbl2 sc1` | 3 | 6 -> 7 | neither | 0.00 | 102400 / 102400 | 4588 |
| `buffer_wbl2 sc1` | 2 | 0 -> 1 | neither | 0.00 | 102400 / 102400 | 6024 |
| `buffer_wbl2 sc1` | 2 | 2 -> 3 | the writer's | 100.00 | 0 / 102400 | 6024 |
| `buffer_wbl2 sc1` | 2 | 4 -> 5 | neither | 0.00 | 102400 / 102400 | 6024 |
| `buffer_wbl2 sc1` | 2 | 6 -> 7 | neither | 0.00 | 102400 / 102400 | 6024 |
| none | - | 0 -> 1 | - | 0.00 | 102400 / 102400 | 0 |
| none | - | 2 -> 3 | - | 0.00 | 102400 / 102400 | 0 |
| none | - | 4 -> 5 | - | 0.00 | 102400 / 102400 | 0 |
| none | - | 6 -> 7 | - | 0.00 | 102400 / 102400 | 0 |
