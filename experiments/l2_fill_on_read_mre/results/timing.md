HIP_VISIBLE_DEVICES=3

Chase loads: sc0 nt (.cg). One wave, 8192 lines (1 MiB), median of 10 launches after one warm-up, a 1 GiB read before each, CUDA events around the kernel. Per load: kernel time with one chase minus with none (first chase), two minus one (second chase), divided by 8192.

| case | kernel us, 0 / 1 / 2 chases | first chase ns per load | second chase ns per load |
|---|---|---|---|
| plain miss | 6 / 3028 / 6044 | 369 | 368 |
| L2 hit (primed, not written) | 23 / 738 / 1459 | 87 | 88 |
| written into L2 | 29 / 3064 / 3748 | 370 | 83 |
| primed, then written | 48 / 752 / 1450 | 86 | 85 |
