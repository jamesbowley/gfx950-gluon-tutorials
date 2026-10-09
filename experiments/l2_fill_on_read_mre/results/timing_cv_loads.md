HIP_VISIBLE_DEVICES=3

Chase loads: sc0 sc1 (.cv, cache volatile). One wave, 8192 lines (1 MiB), median of 10 launches after one warm-up, a 1 GiB read before each, CUDA events around the kernel. Per load: kernel time with one chase minus with none (first chase), two minus one (second chase), divided by 8192.

| case | kernel us, 0 / 1 / 2 chases | first chase ns per load | second chase ns per load |
|---|---|---|---|
| plain miss | 6 / 3013 / 3723 | 367 | 87 |
| L2 hit (primed, not written) | 24 / 741 / 1450 | 88 | 87 |
| written into L2 | 29 / 3054 / 3779 | 369 | 89 |
| primed, then written | 49 / 753 / 1465 | 86 | 87 |
