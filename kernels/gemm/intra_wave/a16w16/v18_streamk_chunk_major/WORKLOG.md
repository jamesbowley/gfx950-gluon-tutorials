# v18_streamk_chunk_major work log

One entry per change: what changed and what it did to performance and spills. Results come
from `HIP_VISIBLE_DEVICES=2 python regress.py ...` (bf16 unless noted), run from `a16w16/`
under the published v10+ stack: Triton v2.2 + deepbind shim on `PYTHONPATH`,
`LLVM_PASS_PLUGIN_PATH=plugins/llir_scheduler/libLlirSched.so`, `TRITON_AMDGCNAS_PLUGIN=1`.
Raw tables are in `results/`.

## 1. Chunk-major mapping with a rotating owner (2026-10-01)

v17 (after its XCD grouping fix) with one change: which program runs which chunk of which
stream-K tile. The tile-aligned split, the chunk boundaries and the tile-to-XCD order stay as
in v17.

**v17 (tile-major).** Tile `t`'s `n` chunks are consecutive spids, so an XCD (32 spids) holds a
few whole tiles. Its programs read up to 32 different k-ranges at once.

**v18 (chunk-major).**
- Work item `w = j * STREAMK_TILES + t` (chunk `j` of tile `t`) goes to spid `w`
  (`streamk_work_item`); the extra tiles' last chunk comes after all the others.
- So an XCD holds the same chunk index of neighbouring tiles, and its programs read the same
  k-steps at the same time:
  - 16 tiles x 16 chunks: XCD `x` holds chunks `2x` and `2x+1` of all 16 tiles;
  - 32 x 8: XCD `j` holds chunk `j` of all 32;
  - one chunk per tile: the same as v17.
- **Rotating owner:** the owner is chunk `t % n`, not chunk 0. Otherwise every owner would sit
  on the XCD that holds chunk 0, and every fixup would read through that XCD's link. Any chunk
  can own a tile, because each program runs one segment: owners wait only on contributors,
  and contributors never wait.
- **Peers are found by formula,** `streamk_peer(i) = ((owner_j + 1 + i) % n) * STREAMK_TILES + t`,
  in the flag poll, the partial reads and the flag reset, instead of as the next spids.

**Correctness and registers.**
- `check_streamk.py`: all 36 configurations pass, plus the capped shapes 256x65792x8192 and
  256x67328x4096, with no amdgcnas fallbacks (`results/check_streamk.md`).
- The build matches v17's: no VGPR spills, 4-5 SGPR spills.
- Regression shapes: within 1.3% of v17.
- Before any GPU run, a pure-Python check confirmed that the decode gives every program a unique
  work item and every tile exactly one owner, whose peers are exactly its other chunks.

### Results

TFLOPS (`results/streamk_shapes.md`, `results/streamk_shapes_split_order.md`,
`results/streamk_shapes_bias.md`); each column's deltas are against v17 in the same run.

| shape | programs per tile | v13 (v9) | v17 (v9) | v18 (v9) | v18 vs v17 (v9) | v18 vs v17 (split) | v18 vs v17 (bias) |
|---|---|---|---|---|---|---|---|
| 4352x4096x8192 | 16 | 1033 | 1214 | 1187 | -2.2% | -1.7% | -3.0% |
| 4352x4352x8192 | 7-8 | 1077 | 1310 | 1333 | +1.7% | +1.6% | +1.9% |
| 4352x4096x4096 | 16 | 973 | 953 | 925 | -2.9% | -2.0% | -2.9% |
| 4352x4096x16384 | 16 | 898 | 1215 | 1249 | +2.7% | -5.4% | +0.0% |
| 3328x5120x8192 | 64 | 1004 | 798 | 793 | -0.6% | -0.1% | -0.7% |
| 4096x6144x8192 | 2 | 1326 | 1505 | 1517 | +0.8% | +0.5% | -0.8% |
| 3840x4096x8192 | 1-2 | 1419 | 1471 | 1477 | +0.4% | -0.3% | +0.6% |
| 8192x8448x8192 | 8 | 1462 | 1571 | 1552 | -1.2% | +0.0% | -1.1% |
| 8192x7936x8192 | 1-2 | 1595 | 1601 | 1604 | +0.2% | +0.2% | +0.1% |

rocprof with cold caches (`results/rocprof.md`): against v17, -3.4% on 4352x4096x8192, +0.9% on
4352x4352x8192, -2.9% on 3328x5120x8192 and +1.1% on 3840x4096x8192.

**Rotating owner** (`results/owner_rotation.md`, a scratch build with the owner fixed at chunk
0): the rotation is worth +2-8% (+8.3% on 4352x4352x8192). With chunk 0 as owner, all owners
would share one or two XCDs.

What the numbers say:

1. **Chunk-major is neutral on these shapes.** Every difference from v17 is within ±3% and
   changes sign between runs, apart from a small, consistent loss on the 16-programs-per-tile
   shapes (4352x4096 at K=4096 and K=8192, -2-3%).
2. **The L2 counters show why** (`results/rocprof.md`): L2 hit rate and memory reads are the
   same as v17 to within 1 point and 3%. Two reasons:
   - The stream-K phase is a small part of these kernels' A/B traffic. On 4352x4096x8192 the
     16 stream-K tiles are 6% of the tiles, and each program spends 8 k-steps loading against
     a tail dominated by collecting partials.
   - Under v9 order the leftover tiles are scattered in vertical pairs that share a B column.
     Tile-major v17 already puts each pair on one XCD, so chunk-major has no further sharing
     to find.

   Even in split order, where all 16 leftover tiles of 4352x4096x8192 share one A row,
   chunk-major gains nothing.
3. **The owner must not be concentrated.** A fixed chunk-0 owner costs up to 8%. That matters
   for any mapping that groups programs by chunk.

**Status.** Kept as a version for the record; it is not better than v17. For a two-tile or
large-K variant, where the stream-K phase carries more of the traffic, it may be worth
revisiting.

### Next

- The remaining one-tile losses are collection-bound: 3328x5120x8192 (63 serial reads per owner)
  and K=4096. After the XCD grouping fix, two-tile on v15/v16's kernel reaches 1136-1160 TFLOPS on
  3328x5120x8192, 15-17% above v13 (v15 WORKLOG entry 3). Its offsets there drift by only 2
  k-steps per program, which is the case the reversed two-tile order (section 8 of the
  explainer) makes general.
- So the next candidate is the reversed two-tile order, on v17's mapping and pipeline, compared
  against v17 on all shapes. Capping the programs per tile is the fallback.
