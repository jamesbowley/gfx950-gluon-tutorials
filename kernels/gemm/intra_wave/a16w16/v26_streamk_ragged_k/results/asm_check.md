# v26 asm check

4096x8192xK, GEMM_TILE_LIMIT = 256 + S. Aligned K: v26 against v25 (kernel name replaced, comments and directives dropped). K - 16, K - 112: v26 with EDGE_K; forced: also STREAMK_FORCE_EDGE=1.

| K | S | configuration | v25 VGPRs / spills | v26 VGPRs / spills | instruction lines changed | K - 16 | K - 112 | K - 112 forced | result |
|---|---|---|---|---|---|---|---|---|---|
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128 | 448 / 0 | 448 / 0 | 0 | 460 / 0 | 460 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100 | 448 / 0 | 448 / 0 | 0 | 460 / 0 | 460 / 0 | 492 / 0 | identical |
| 8192 | 38 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=152 | 452 / 0 | 452 / 0 | 0 | 464 / 0 | 464 / 0 | 492 / 0 | identical |
| 4096 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs | 444 / 0 | 444 / 0 | 0 | 460 / 0 | 460 / 0 | 492 / 0 | identical |
| 4096 | 16 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=64, PARTIALS=bf16 | 452 / 0 | 452 / 0 | 0 | 460 / 0 | 460 / 0 | 492 / 0 | identical |
| 8192 | 24 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=144, RS_READERS=all | 452 / 0 | 452 / 0 | 0 | 464 / 0 | 460 / 0 | 492 / 0 | identical |
| 8192 | 16 | POLICY=two_tile | 512 / 15 | 512 / 6 | 4426 | 512 / 9 | 512 / 9 | 512 / 14 | changed (end-to-end, entry 3) |
| 16384 | 192 | POLICY=spread | 512 / 15 | 512 / 6 | 4387 | 512 / 7 | 512 / 5 | 512 / 22 | changed (end-to-end, entry 3) |
| 8192 | 16 | POLICY=dp | 440 / 0 | 440 / 0 | 0 | 448 / 0 | 452 / 0 | 488 / 0 | identical |
| 8192 | 16 | (defaults) | 448 / 0 | 448 / 0 | 0 | 460 / 0 | 460 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | 480 / 0 | 480 / 0 | 0 | 480 / 0 | 480 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | 480 / 0 | 480 / 0 | 0 | 480 / 0 | 480 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=tile | 452 / 0 | 452 / 0 | 0 | 456 / 0 | 460 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=tile | 452 / 0 | 452 / 0 | 0 | 456 / 0 | 460 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | 484 / 0 | 484 / 0 | 0 | 488 / 0 | 488 / 0 | 500 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | 480 / 0 | 480 / 0 | 0 | 480 / 0 | 480 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=chunk | 452 / 0 | 452 / 0 | 0 | 468 / 0 | 464 / 0 | 496 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=chunk | 452 / 0 | 452 / 0 | 0 | 456 / 0 | 460 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | 448 / 0 | 448 / 0 | 0 | 460 / 0 | 460 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | 448 / 0 | 448 / 0 | 0 | 460 / 0 | 460 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=tile | 448 / 0 | 448 / 0 | 0 | 460 / 0 | 460 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=tile | 444 / 0 | 444 / 0 | 0 | 460 / 0 | 460 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | 448 / 0 | 448 / 0 | 0 | 460 / 0 | 460 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | 448 / 0 | 448 / 0 | 0 | 460 / 0 | 460 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=chunk | 448 / 0 | 448 / 0 | 0 | 460 / 0 | 460 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=chunk | 444 / 0 | 444 / 0 | 0 | 460 / 0 | 460 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | 448 / 0 | 448 / 0 | 0 | 460 / 0 | 460 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | 448 / 0 | 448 / 0 | 0 | 460 / 0 | 460 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | 448 / 0 | 448 / 0 | 0 | 460 / 0 | 460 / 0 | 492 / 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | 448 / 0 | 448 / 0 | 0 | 460 / 0 | 460 / 0 | 492 / 0 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, EXCHANGE=l2 | 448 / 0 | 448 / 0 | 0 | 460 / 0 | 460 / 0 | 492 / 0 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=64, PARTIALS=fp32, EXCHANGE=l2 | 448 / 0 | 448 / 0 | 0 | 460 / 0 | 460 / 0 | 492 / 0 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, EXCHANGE=l2 | 444 / 0 | 444 / 0 | 0 | 460 / 0 | 460 / 0 | 492 / 0 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=64, PARTIALS=bf16_pairs, EXCHANGE=l2 | 448 / 0 | 448 / 0 | 0 | 460 / 0 | 460 / 0 | 492 / 0 | identical |

PASS
