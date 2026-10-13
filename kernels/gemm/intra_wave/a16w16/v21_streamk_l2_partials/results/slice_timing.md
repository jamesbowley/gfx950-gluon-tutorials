Medians in us over the split programs and launches. reads: from the barrier to the slice's reads completing; bias + C: the rest of the slice; in MAC: other split programs still in their MAC loop when this program's slice starts. The probe loop writes no C, so its results are wrong by design.

| shape | S | n | partials | slice code | MAC | store | barrier | reads | bias + C | slice | in MAC (of programs) | wrong |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4352x4096x8192 | 16 | 8 | mem | GEMM slice | 20.9 | 5.4 | 4.3 | 8.2 | 1.1 | 9.3 | 32 (of 128) | 0 |
| 4352x4096x8192 | 16 | 8 | mem | probe loop | 20.6 | 5.4 | 4.4 | 4.0 | 0.6 | 4.6 | 32 (of 128) | 11 |
| 4352x4096x8192 | 16 | 8 | l2 | GEMM slice | 21.2 | 3.4 | 0.8 | 8.9 | 1.0 | 9.8 | 32 (of 128) | 0 |
| 4352x4096x8192 | 16 | 8 | l2 | probe loop | 21.2 | 3.4 | 0.9 | 6.7 | 0.6 | 7.2 | 32 (of 128) | 11 |
| 3072x5632x8192 | 8 | 16 | mem | GEMM slice | 14.1 | 5.6 | 4.1 | 8.4 | 0.7 | 9.1 | 1 (of 128) | 0 |
| 3072x5632x8192 | 8 | 16 | mem | probe loop | 13.6 | 5.8 | 4.4 | 4.4 | 0.4 | 4.9 | 0 (of 128) | 11 |
| 3072x5632x8192 | 8 | 16 | l2 | GEMM slice | 14.0 | 3.5 | 1.0 | 8.9 | 0.6 | 9.5 | 32 (of 128) | 0 |
| 3072x5632x8192 | 8 | 16 | l2 | probe loop | 14.1 | 3.5 | 1.1 | 6.1 | 0.4 | 6.6 | 26 (of 128) | 11 |
| 4096x4608x8192 | 32 | 4 | mem | GEMM slice | 38.9 | 4.4 | 5.8 | 7.8 | 2.4 | 10.2 | 0 (of 128) | 0 |
| 4096x4608x8192 | 32 | 4 | mem | probe loop | 38.4 | 4.7 | 5.4 | 3.9 | 0.9 | 4.8 | 0 (of 128) | 11 |
| 4096x4608x8192 | 32 | 4 | l2 | GEMM slice | 38.6 | 3.4 | 0.9 | 8.2 | 2.9 | 11.3 | 2 (of 128) | 0 |
| 4096x4608x8192 | 32 | 4 | l2 | probe loop | 38.4 | 3.4 | 0.8 | 6.8 | 1.0 | 7.8 | 32 (of 128) | 11 |

## Why the GEMM slice's reads are slower: lane order

Both read loops issue the same `buffer_load_dwordx4` count with the same waits. They differ in which
address each lane gets. In the GEMM slice's layout (`BlockedLayout([1,1,4],[1,16,4],[1,2,2],[2,1,0])`
through `in_blk`) lane `l` reads float `4*(l//4) + 64*(l%4)` of the 1 KiB a wave covers, so lanes
0-3 are 256 bytes apart, in four different 128-byte lines. The probe's lanes 0-3 read 64
contiguous bytes of one line. The L1 takes 4 lanes per cycle and does not merge across cycles, so
the GEMM slice looks up each line 8 times (16 bytes each) where the probe looks it up twice; and lines
256 bytes apart share a tag-RAM set, so lookups in the same cycle conflict.

rocprofv3 counters, 4352x4096x8192, `mem`, kernel medians (whole kernel, so the MAC loop's
traffic is in both; the difference is the slice):

| counter | GEMM slice | probe loop | difference |
|---|---|---|---|
| TCP_TCC_READ_REQ_sum (L1 to L2 reads) | 18.482M | 18.482M | 0 |
| TCP_TOTAL_CACHE_ACCESSES_sum | 39.651M | 38.013M | +1.64M |
| TCP_READ_TAGCONFLICT_STALL_CYCLES_sum | 0.262M | 0 | +0.262M |
| TCP_TCP_TA_ADDR_STALL_CYCLES_sum | 1.769M | 1.007M | +0.76M |

- L2 sees the same requests: the L1 merges the repeated lookups, so no extra traffic leaves the CU.
- The extra L1 accesses match the prediction: 32 MiB of partials is 262K lines, times 6 extra
  lookups per line, is 1.57M.
- Tag-conflict stalls: 262K, one per partial line.
- Address stalls: 0.76M cycles over the 128 slice CUs is about 5.9K cycles each, about 2.5 us at
  2.4 GHz, most of the 4 us gap.

## After the fix: rows first in lane order

`L3` order `[1, 2, 0]` instead of `[2, 1, 0]`: lane `l` gets row `l % 16` and columns
`4 (l // 16) .. + 3`, which `in_blk` puts at `4 l .. 4 l + 3`, the probe's addresses. Same run
(MAC times are higher than above on every row, the GPU's clock that day; compare within the table):

| shape | S | n | partials | slice code | MAC | store | barrier | reads | bias + C | slice | wrong |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4352x4096x8192 | 16 | 8 | mem | GEMM slice | 24.5 | 4.3 | 4.3 | 4.6 | 1.6 | 6.2 | 0 |
| 4352x4096x8192 | 16 | 8 | mem | probe loop | 24.7 | 4.3 | 4.6 | 4.3 | 0.6 | 4.9 | 11 |
| 4352x4096x8192 | 16 | 8 | l2 | GEMM slice | 24.4 | 3.2 | 0.9 | 6.3 | 1.5 | 7.8 | 0 |
| 4352x4096x8192 | 16 | 8 | l2 | probe loop | 24.4 | 3.2 | 1.2 | 7.4 | 0.6 | 8.0 | 11 |
| 3072x5632x8192 | 8 | 16 | mem | GEMM slice | 14.6 | 4.4 | 4.1 | 5.0 | 1.0 | 6.1 | 0 |
| 3072x5632x8192 | 8 | 16 | mem | probe loop | 14.8 | 4.5 | 4.3 | 4.4 | 0.4 | 4.9 | 11 |
| 3072x5632x8192 | 8 | 16 | l2 | GEMM slice | 14.8 | 3.2 | 1.0 | 6.6 | 1.0 | 7.5 | 0 |
| 3072x5632x8192 | 8 | 16 | l2 | probe loop | 15.0 | 3.2 | 1.1 | 6.3 | 0.5 | 6.8 | 11 |
| 4096x4608x8192 | 32 | 4 | mem | GEMM slice | 47.0 | 4.4 | 5.2 | 4.5 | 3.7 | 8.1 | 0 |
| 4096x4608x8192 | 32 | 4 | mem | probe loop | 47.5 | 4.2 | 5.5 | 4.0 | 0.9 | 4.9 | 11 |
| 4096x4608x8192 | 32 | 4 | l2 | GEMM slice | 47.2 | 3.4 | 0.9 | 6.1 | 3.9 | 10.1 | 0 |
| 4096x4608x8192 | 32 | 4 | l2 | probe loop | 47.4 | 3.4 | 0.9 | 6.9 | 0.9 | 7.9 | 11 |

- Reads: 7.8-8.4 us down to 4.5-5.0 under `mem`, within 0.6 us of the probe loop; 8.2-8.9 down
  to 6.1-6.6 under `l2`.
- Bias + C store: up by 0.3-1.3 us, because each 4-lane group now writes 4 C rows (8 bytes each)
  instead of 32 bytes of one row.
- Slice under `mem`: 9.1-10.2 us down to 6.1-8.1.
- Tried and dropped: loading rows first and `gl.convert_layout` to columns first for the bias and C
  store. The conversion goes through LDS and bias + C rose to 2.4-10.5 us.
- Kernel times: [rocprof_lane_order.md](rocprof_lane_order.md).
