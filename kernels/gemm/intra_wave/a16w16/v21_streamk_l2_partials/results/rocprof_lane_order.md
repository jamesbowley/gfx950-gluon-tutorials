Reduce-scatter slice with rows first in lane order (`L3` order `[1, 2, 0]`, the fix), against
columns first (`[2, 1, 0]`, before), in v20 and v21. GPU 1, published stack, `rocprof_check.py
--no-l2` (cold kernel time, median of 1000 launches, us). Two runs: run 1 timed the fixed kernels
first, run 2 the old ones first. v19 has no slice and did not change; it shows the run-to-run
drift (within 1.4 us).

| shape | version | run 1 before | run 1 after | run 2 before | run 2 after | gain (mean) |
|---|---|---|---|---|---|---|
| 3072x5632x8192 | v19 | | 233.6 | 233.0 | 233.6 | |
| 3072x5632x8192 | v20 | 232.3 | 229.2 | 231.5 | 229.3 | 2.7 |
| 3072x5632x8192 | v21 mem | 232.4 | 227.8 | 231.9 | 229.6 | 3.5 |
| 3072x5632x8192 | v21 l2 | 229.0 | 226.6 | 228.7 | 228.1 | 1.5 |
| 4352x4096x8192 | v19 | | 243.0 | 242.2 | 241.8 | |
| 4352x4096x8192 | v20 | 238.5 | 235.3 | 240.5 | 236.5 | 3.6 |
| 4352x4096x8192 | v21 mem | 238.8 | 236.6 | 238.9 | 235.3 | 2.9 |
| 4352x4096x8192 | v21 l2 | 237.2 | 235.7 | 237.4 | 235.7 | 1.6 |
| 4096x4608x8192 | v19 | | 256.0 | 255.0 | 256.3 | |
| 4096x4608x8192 | v20 | 258.8 | 257.2 | 259.5 | 257.4 | 1.9 |
| 4096x4608x8192 | v21 mem | 260.4 | 257.4 | 260.1 | 257.5 | 2.8 |
| 4096x4608x8192 | v21 l2 | 255.7 | 254.4 | 256.0 | 254.9 | 1.2 |
| 3328x5120x8192 | v19 | | 230.5 | 230.8 | 230.2 | |
| 3328x5120x8192 | v20 | 223.2 | 218.5 | 223.2 | 219.0 | 4.5 |
| 3328x5120x8192 | v21 mem | 222.9 | 219.1 | 222.9 | 219.4 | 3.7 |
| 3328x5120x8192 | v21 l2 | 223.9 | 221.2 | 223.9 | 223.4 | 1.6 |
| 4352x4352x8192 | v19 | | 257.5 | 257.3 | 258.4 | |
| 4352x4352x8192 | v20 | 258.8 | 258.8 | 259.0 | 259.0 | 0.0 |
| 4352x4352x8192 | v21 mem | 258.9 | 258.6 | 258.5 | 259.1 | -0.2 |
| 4352x4352x8192 | v21 l2 | 257.8 | 257.1 | 258.0 | 257.5 | 0.6 |
| 1536x11008x8192 | v19 | | 226.6 | 228.7 | 227.3 | |
| 1536x11008x8192 | v20 | 223.0 | 220.4 | 223.2 | 219.0 | 3.4 |
| 1536x11008x8192 | v21 mem | 223.2 | 219.9 | 223.1 | 219.4 | 3.5 |
| 1536x11008x8192 | v21 l2 | 224.0 | 222.6 | 224.2 | 220.8 | 2.4 |
| 4352x4096x4096 | v19 | | 135.3 | 135.4 | 134.9 | |
| 4352x4096x4096 | v20 | 135.4 | 132.5 | 135.1 | 132.8 | 2.6 |
| 4352x4096x4096 | v21 mem | 135.3 | 132.8 | 137.0 | 132.4 | 3.6 |
| 4352x4096x4096 | v21 l2 | 133.2 | 131.3 | 133.6 | 131.1 | 2.2 |

- v20 and v21 `mem` gain 2-4.5 us (1-3%) wherever the slice runs. 4352x4352x8192 (S = 33) uses
  the owner fixup, which has no slice, so it does not move.
- `l2` gains less (0.6-2.4 us). Its slice reads are limited by the L2 fill on the first read of
  each written line (see experiments/l2_memsys), not by the L1, so `l2` and `mem` are now within
  a few us of each other either way.
- v20 against v19 (run 2): v20 was ahead on five shapes by 0.3-7.6 us and is now ahead on the
  same five by 2.1-11.2 us; on 4096x4608 and 4352x4352 it is still 0.6-1.1 us behind.
