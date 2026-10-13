# E1: same addresses or a property of the buffer

`python place.py e1 --n 10 --seed {0,1}`: v9, 4096x4096x1024, bf16, GPU 2, LLIR + amdgcnas. Ten
caching-allocator C buffers are classified by their single-buffer back-to-back time; F1, F2 are
the two fastest and S1, S2 the two slowest. Each pattern is a graph of 400 GEMMs that writes
the listed buffers in turn, replayed 100 times; us is the median per GEMM, MHz and W the median
hwmon sclk and power while it replays.

Classified (us), seed 0: 28.27 28.38 28.40 28.42 28.54 | 29.44 29.45 29.47 29.49 29.53.
Seed 1: 28.21 28.32 28.64 | 29.45 29.47 29.47 29.50 29.50 29.52 29.52.

| pattern | seed 0 us | MHz | W | seed 1 us | MHz | W |
|---|---|---|---|---|---|---|
| F1 | 28.33 | 2046 | 1392 | 28.52 | 2034 | 1392 |
| F2 | 28.43 | 2039 | 1388 | 28.60 | 2038 | 1390 |
| S1 | 29.45 | 2001 | 1336 | 29.49 | 2011 | 1335 |
| S2 | 29.44 | 2004 | 1338 | 29.46 | 2005 | 1335 |
| F1,F2 | 28.43 | 2034 | 1396 | 28.46 | 2028 | 1394 |
| F1,F1,F2,F2 | 28.38 | 2015 | 1396 | 28.53 | 2026 | 1391 |
| F1 x4, F2 x4 | 28.40 | 2017 | 1393 | 28.57 | 2031 | 1388 |
| F1 x16, F2 x16 | 28.43 | 2017 | 1396 | 28.50 | 2027 | 1388 |
| F1,S1 | 29.40 | 2009 | 1339 | 29.41 | 2010 | 1342 |
| S1,S2 | 29.44 | 2003 | 1343 | 29.43 | 2008 | 1337 |
| S1,S1,S2,S2 | 29.41 | 2007 | 1341 | 29.42 | 2009 | 1340 |

**Conclusion.** Fast is a property of the buffer: two fast buffers stay fast however they
alternate, so consecutive kernels need not write the same lines. Alternating a fast and a
slow buffer is as slow as the slow one alone (29.40 us, not the 28.9 us average), so the
fast buffer's GEMMs run slow too: the slow state persists across kernels. That is why rotating
two or more random copies always measured slow. The slow mode also runs at a lower clock and
about 50 W less power, which is not what a power limit would do.
