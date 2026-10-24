# v26 asm check

4096x8192xK, GEMM_TILE_LIMIT = 256 + S. Aligned K: v26 against v25 (kernel name replaced, comments and directives dropped). K - 16, K - 112: v26 with EDGE_K; forced: also STREAMK_FORCE_EDGE=1.

| K | S | configuration | v25 VGPRs / spills | v26 VGPRs / spills | instruction lines changed | K - 16 | K - 112 | K - 112 forced | result |
|---|---|---|---|---|---|---|---|---|---|
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128 | 472 / 0 | 472 / 0 | 0 | 480 / 0 | 484 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100 | 468 / 0 | 468 / 0 | 0 | 480 / 0 | 480 / 0 | 492 / 0 | identical |
| 8192 | 38 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=152 | 472 / 0 | 472 / 0 | 0 | 480 / 0 | 484 / 0 | 492 / 0 | identical |
| 4096 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs | 468 / 0 | 468 / 0 | 0 | 480 / 0 | 480 / 0 | 492 / 0 | identical |
| 4096 | 16 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=64, PARTIALS=bf16 | 472 / 0 | 472 / 0 | 0 | 480 / 0 | 484 / 0 | 492 / 0 | identical |
| 8192 | 24 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=144, RS_READERS=all | 472 / 0 | 472 / 0 | 0 | 480 / 0 | 484 / 0 | 492 / 0 | identical |
| 8192 | 16 | POLICY=two_tile | 512 / 15 | 512 / 6 | 6370 | 512 / 9 | 512 / 9 | 512 / 14 | changed (end-to-end, entry 3) |
| 16384 | 192 | POLICY=spread | 512 / 15 | 512 / 6 | 6758 | 512 / 7 | 512 / 5 | 512 / 22 | changed (end-to-end, entry 3) |
| 8192 | 16 | POLICY=dp | 464 / 0 | 464 / 0 | 0 | 472 / 0 | 472 / 0 | 488 / 0 | identical |
| 8192 | 16 | (defaults) | 468 / 0 | 468 / 0 | 0 | 480 / 0 | 480 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | 480 / 0 | 480 / 0 | 0 | 480 / 0 | 484 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | 480 / 0 | 480 / 0 | 0 | 480 / 0 | 484 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=tile | 472 / 0 | 472 / 0 | 0 | 480 / 0 | 484 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=tile | 472 / 0 | 472 / 0 | 0 | 480 / 0 | 484 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | 484 / 0 | 484 / 0 | 0 | 488 / 0 | 492 / 0 | 500 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | 480 / 0 | 480 / 0 | 0 | 480 / 0 | 484 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=chunk | 472 / 0 | 472 / 0 | 0 | 480 / 0 | 488 / 0 | 496 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=chunk | 472 / 0 | 472 / 0 | 0 | 480 / 0 | 484 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | 472 / 0 | 472 / 0 | 0 | 480 / 0 | 484 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | 468 / 0 | 468 / 0 | 0 | 480 / 0 | 480 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=tile | 472 / 0 | 472 / 0 | 0 | 480 / 0 | 484 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=tile | 468 / 0 | 468 / 0 | 0 | 480 / 0 | 480 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | 472 / 0 | 472 / 0 | 0 | 480 / 0 | 484 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | 468 / 0 | 468 / 0 | 0 | 480 / 0 | 480 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=chunk | 472 / 0 | 472 / 0 | 0 | 480 / 0 | 484 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=chunk | 468 / 0 | 468 / 0 | 0 | 480 / 0 | 480 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | 472 / 0 | 472 / 0 | 0 | 476 / 0 | 484 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | 472 / 0 | 472 / 0 | 0 | 476 / 0 | 484 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | 472 / 0 | 472 / 0 | 0 | 476 / 0 | 484 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | 472 / 0 | 472 / 0 | 0 | 476 / 0 | 484 / 0 | 492 / 0 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, EXCHANGE=l2 | 468 / 0 | 468 / 0 | 0 | 480 / 0 | 480 / 0 | 492 / 0 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=64, PARTIALS=fp32, EXCHANGE=l2 | 468 / 0 | 468 / 0 | 0 | 480 / 0 | 480 / 0 | 492 / 0 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, EXCHANGE=l2 | 468 / 0 | 468 / 0 | 0 | 480 / 0 | 480 / 0 | 492 / 0 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=64, PARTIALS=bf16_pairs, EXCHANGE=l2 | 468 / 0 | 468 / 0 | 0 | 480 / 0 | 480 / 0 | 492 / 0 | identical |

PASS
