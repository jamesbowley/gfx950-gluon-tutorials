HIP_VISIBLE_DEVICES=3

Per 128-byte line of the 1 MiB buffer (8192 lines), median of 10 launches after one warm-up, a 1 GiB read before each. 'Before the chase' is the kernel without the chase (prime and write only); 'chase' is the kernel with one chase minus that.

| case | phase | L2 hits | L2 misses | memory reads | of which DRAM |
|---|---|---|---|---|---|
| plain miss | before the chase | 0.00 | 0.00 | 0.00 | 0.00 |
| | chase | 0.00 | 1.00 | 1.00 | 1.00 |
| L2 hit (primed, not written) | before the chase | 0.00 | 1.00 | 1.00 | 1.00 |
| | chase | 1.00 | 0.00 | 0.00 | 0.00 |
| written into L2 | before the chase | 1.00 | 1.00 | 0.00 | 0.00 |
| | chase | 1.00 | 0.00 | 1.00 | 1.00 |
| primed, then written | before the chase | 2.00 | 1.00 | 1.00 | 1.00 |
| | chase | 1.00 | 0.00 | 0.00 | 0.00 |
