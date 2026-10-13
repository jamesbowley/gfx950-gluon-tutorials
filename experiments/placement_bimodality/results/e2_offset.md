# E2: C's offset inside one 1 GB buffer

v9, 4096x4096x1024, bf16, GPU 2, LLIR + amdgcnas. One 1 GB `hipMalloc` buffer holds A, B (8 MB
each, contiguous) and C (32 MB) at chosen offsets; each point is a back-to-back graph of 400
GEMMs, median of 10 replays. Each run allocates a new buffer, so the phase of the pattern can
shift by a few MB between runs. Raw tables: `e2_raw.txt` (16 KB steps to 512 KB, then 8 MB
steps), `e2_c_sweep.txt`, `e2_ab_sweep_cfast.txt`, `e2_ab_sweep_cslow.txt`,
`e2_boundary_lo.txt`, `e2_boundary_hi.txt`.

## C swept, A/B fixed at 900 MB (`sweep --what c --step 2`)

| C offset (MB) | us |
|---|---|
| 0 - 98 | 28.2 - 28.9 (fast; a few points up to 29.2) |
| 100 - 254 | 29.37 - 29.69 (slow) |
| 256 - 352 | 28.3 - 29.3 (fast) |
| 354 - 510 | 29.35 - 29.67 (slow) |
| 512 | 28.53 |

The pattern repeats every 256 MB (four periods in the coarse sweep, `e2_raw.txt`). C is 32 MB,
so it is fast only when all of it lies in the first half of a period: at offset 254 MB, where
just 2 MB of C (254 - 256 MB) lies in the second half, it is as slow as anywhere (29.40 us).

## The boundary at 0.25 MB resolution (another buffer)

| C offset (MB) | 96 | 96.25 | 96.5 | 97 | 98 | 100 |
|---|---|---|---|---|---|---|
| us | 28.29 | 28.69 | 29.47 | 29.50 | 29.13 | 29.40 |

C turns slow when its end passes 128.25-128.5 MB of the buffer, so in this buffer the slow half
starts at 128 MB; in the 2 MB sweep's buffer it started about 3 MB later. Just above 256 MB
the values are mixed (29.06, 28.85, 29.20, 29.52 at 256 - 256.75 MB) before the fast half,
and a few fast-half points elsewhere sit at 28.9 - 29.3, so the halves are not perfectly clean.

## A/B swept over the buffer (`sweep --what ab --step 8`)

| C | A/B offsets | us |
|---|---|---|
| fast (32 MB) | 64 - 1000 MB | 28.17 - 29.24, median 28.47 |
| slow (160 MB) | 0 - 1000 MB | 29.40 - 29.68 |

**Conclusion.** Within one contiguous allocation the result is set by C's address: a GEMM is
fast only if every line of C lies in the first 128 MB of a 256 MB period, and slow as soon as
any of C falls in the second half. Where A and B sit makes no difference, so it is the writes
that matter. A 256 MB period fixed in the address, not in the buffer, points at a physical
address bit (bit 27 of the 128 MB halves) and an address-to-memory mapping (which HBM stack,
memory-side die or MALL slice serves that half). E3 tests whether it is the physical or the
virtual address.
