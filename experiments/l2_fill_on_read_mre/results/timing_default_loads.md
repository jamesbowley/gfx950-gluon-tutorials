HIP_VISIBLE_DEVICES=3

Chase loads: default. One wave, 8192 lines (1 MiB), median of 10 launches after one warm-up, a 1 GiB read before each, CUDA events around the kernel. Per load: kernel time with one chase minus with none (first chase), two minus one (second chase), divided by 8192.

| case | kernel us, 0 / 1 / 2 chases | first chase ns per load | second chase ns per load |
|---|---|---|---|
| plain miss | 6 / 3035 / 3729 | 370 | 85 |
| L2 hit (primed, not written) | 23 / 729 / 1432 | 86 | 86 |
| written into L2 | 30 / 3070 / 3759 | 371 | 84 |
| primed, then written | 49 / 769 / 1474 | 88 | 86 |
