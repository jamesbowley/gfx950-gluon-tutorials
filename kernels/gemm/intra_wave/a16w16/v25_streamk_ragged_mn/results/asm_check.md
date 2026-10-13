# v25 asm check

4096x8192xK, GEMM_TILE_LIMIT = 256 + S. Flags off: v25 against v24 (kernel name replaced, comments and directives dropped). Forced: v25 with STREAMK_FORCE_EDGE=1.

| K | S | configuration | v24 VGPRs / spills | v25 off VGPRs / spills | instruction lines changed | v25 forced VGPRs / spills | result |
|---|---|---|---|---|---|---|---|
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128 | 448 / 0 | 448 / 0 | 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100 | 448 / 0 | 448 / 0 | 0 | 488 / 0 | identical |
| 8192 | 38 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=152 | 452 / 0 | 452 / 0 | 0 | 492 / 0 | identical |
| 4096 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs | 444 / 0 | 444 / 0 | 0 | 488 / 0 | identical |
| 4096 | 16 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=64, PARTIALS=bf16 | 452 / 0 | 452 / 0 | 0 | 492 / 0 | identical |
| 8192 | 24 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=144, RS_READERS=all | 452 / 0 | 452 / 0 | 0 | 492 / 0 | identical |
| 8192 | 16 | POLICY=two_tile | 512 / 15 | 512 / 15 | 0 | 512 / 25 | identical |
| 16384 | 192 | POLICY=spread | 512 / 15 | 512 / 15 | 0 | 512 / 21 | identical |
| 8192 | 16 | POLICY=dp | 440 / 0 | 440 / 0 | 0 | 484 / 0 | identical |
| 8192 | 16 | (defaults) | 448 / 0 | 448 / 0 | 0 | 488 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | 480 / 0 | 480 / 0 | 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | 480 / 0 | 480 / 0 | 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=tile | 452 / 0 | 452 / 0 | 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=tile | 452 / 0 | 452 / 0 | 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | 484 / 0 | 484 / 0 | 0 | 500 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | 480 / 0 | 480 / 0 | 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=chunk | 452 / 0 | 452 / 0 | 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=chunk | 452 / 0 | 452 / 0 | 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | 448 / 0 | 448 / 0 | 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | 448 / 0 | 448 / 0 | 0 | 488 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=tile | 448 / 0 | 448 / 0 | 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=tile | 444 / 0 | 444 / 0 | 0 | 488 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | 448 / 0 | 448 / 0 | 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | 448 / 0 | 448 / 0 | 0 | 488 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=chunk | 448 / 0 | 448 / 0 | 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=chunk | 444 / 0 | 444 / 0 | 0 | 488 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | 448 / 0 | 448 / 0 | 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | 448 / 0 | 448 / 0 | 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | 448 / 0 | 448 / 0 | 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | 448 / 0 | 448 / 0 | 0 | 492 / 0 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, EXCHANGE=l2 | 448 / 0 | 448 / 0 | 0 | 488 / 0 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=64, PARTIALS=fp32, EXCHANGE=l2 | 448 / 0 | 448 / 0 | 0 | 488 / 0 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, EXCHANGE=l2 | 444 / 0 | 444 / 0 | 0 | 488 / 0 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=64, PARTIALS=bf16_pairs, EXCHANGE=l2 | 448 / 0 | 448 / 0 | 0 | 488 / 0 | identical |

PASS
