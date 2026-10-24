# v27 gate: 256x256x64 against v26

4096x8192xK, GEMM_TILE_LIMIT = 256 + S; v26 and v27 (default tile) on the same configuration. Instruction lines (kernel name replaced, comments and directives dropped), VGPRs / AGPRs / spills, and v27's launched grid (must be all 256 programs: the compact launch never applies with full tiles).

| K | S | configuration | build | v26 VGPRs / AGPRs / spills | v27 | instructions | lines changed | grid | result |
|---|---|---|---|---|---|---|---|---|---|
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128 | K | 472 / 256 / 0 | 472 / 256 / 0 | 4932 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128 | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 4992 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128 | K - 112 | 484 / 256 / 0 | 484 / 256 / 0 | 5008 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128 | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 4845 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100 | K | 468 / 256 / 0 | 468 / 256 / 0 | 4944 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100 | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 5026 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100 | K - 112 | 480 / 256 / 0 | 480 / 256 / 0 | 5021 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100 | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 4870 | 0 | 256 | identical |
| 8192 | 38 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=152 | K | 472 / 256 / 0 | 472 / 256 / 0 | 6178 | 0 | 256 | identical |
| 8192 | 38 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=152 | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 6251 | 0 | 256 | identical |
| 8192 | 38 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=152 | K - 112 | 484 / 256 / 0 | 484 / 256 / 0 | 6249 | 0 | 256 | identical |
| 8192 | 38 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=152 | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 6192 | 0 | 256 | identical |
| 4096 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs | K | 468 / 256 / 0 | 468 / 256 / 0 | 5383 | 0 | 256 | identical |
| 4096 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 5450 | 0 | 256 | identical |
| 4096 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs | K - 112 | 480 / 256 / 0 | 480 / 256 / 0 | 5446 | 0 | 256 | identical |
| 4096 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 5296 | 0 | 256 | identical |
| 4096 | 16 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=64, PARTIALS=bf16 | K | 472 / 256 / 0 | 472 / 256 / 0 | 6907 | 0 | 256 | identical |
| 4096 | 16 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=64, PARTIALS=bf16 | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 6981 | 0 | 256 | identical |
| 4096 | 16 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=64, PARTIALS=bf16 | K - 112 | 484 / 256 / 0 | 484 / 256 / 0 | 6979 | 0 | 256 | identical |
| 4096 | 16 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=64, PARTIALS=bf16 | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 6931 | 0 | 256 | identical |
| 8192 | 24 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=144, RS_READERS=all | K | 472 / 256 / 0 | 472 / 256 / 0 | 5913 | 0 | 256 | identical |
| 8192 | 24 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=144, RS_READERS=all | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 5975 | 0 | 256 | identical |
| 8192 | 24 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=144, RS_READERS=all | K - 112 | 484 / 256 / 0 | 484 / 256 / 0 | 5993 | 0 | 256 | identical |
| 8192 | 24 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=144, RS_READERS=all | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 5885 | 0 | 256 | identical |
| 8192 | 16 | POLICY=two_tile | K | 512 / 256 / 6 | 512 / 256 / 6 | 7779 | 0 | 256 | identical |
| 8192 | 16 | POLICY=two_tile | K - 16 | 512 / 256 / 9 | 512 / 256 / 9 | 4657 | 0 | 256 | identical |
| 8192 | 16 | POLICY=two_tile | K - 112 | 512 / 256 / 9 | 512 / 256 / 9 | 4656 | 0 | 256 | identical |
| 8192 | 16 | POLICY=two_tile | K - 112, all edges | 512 / 256 / 14 | 512 / 256 / 14 | 4859 | 0 | 256 | identical |
| 16384 | 192 | POLICY=spread | K | 512 / 256 / 6 | 512 / 256 / 6 | 7797 | 0 | 256 | identical |
| 16384 | 192 | POLICY=spread | K - 16 | 512 / 256 / 7 | 512 / 256 / 7 | 7957 | 0 | 256 | identical |
| 16384 | 192 | POLICY=spread | K - 112 | 512 / 256 / 5 | 512 / 256 / 5 | 7938 | 0 | 256 | identical |
| 16384 | 192 | POLICY=spread | K - 112, all edges | 512 / 256 / 22 | 512 / 256 / 22 | 7918 | 0 | 256 | identical |
| 8192 | 16 | POLICY=dp | K | 464 / 256 / 0 | 464 / 256 / 0 | 3337 | 0 | 256 | identical |
| 8192 | 16 | POLICY=dp | K - 16 | 472 / 256 / 0 | 472 / 256 / 0 | 3361 | 0 | 256 | identical |
| 8192 | 16 | POLICY=dp | K - 112 | 472 / 256 / 0 | 472 / 256 / 0 | 3371 | 0 | 256 | identical |
| 8192 | 16 | POLICY=dp | K - 112, all edges | 488 / 256 / 0 | 488 / 256 / 0 | 3120 | 0 | 256 | identical |
| 8192 | 16 | (defaults) | K | 468 / 256 / 0 | 468 / 256 / 0 | 4943 | 0 | 256 | identical |
| 8192 | 16 | (defaults) | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 5023 | 0 | 256 | identical |
| 8192 | 16 | (defaults) | K - 112 | 480 / 256 / 0 | 480 / 256 / 0 | 5019 | 0 | 256 | identical |
| 8192 | 16 | (defaults) | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 4870 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | K | 480 / 256 / 0 | 480 / 256 / 0 | 6865 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 6939 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | K - 112 | 484 / 256 / 0 | 484 / 256 / 0 | 6938 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 6885 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | K | 480 / 256 / 0 | 480 / 256 / 0 | 6875 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 6953 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | K - 112 | 484 / 256 / 0 | 484 / 256 / 0 | 6952 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 6906 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=tile | K | 472 / 256 / 0 | 472 / 256 / 0 | 6801 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=tile | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 6867 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=tile | K - 112 | 484 / 256 / 0 | 484 / 256 / 0 | 6866 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=tile | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 6809 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=tile | K | 472 / 256 / 0 | 472 / 256 / 0 | 6811 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=tile | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 6889 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=tile | K - 112 | 484 / 256 / 0 | 484 / 256 / 0 | 6888 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=tile | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 6838 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | K | 484 / 256 / 0 | 484 / 256 / 0 | 7478 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | K - 16 | 488 / 256 / 0 | 488 / 256 / 0 | 7581 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | K - 112 | 492 / 256 / 0 | 492 / 256 / 0 | 7554 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | K - 112, all edges | 500 / 256 / 0 | 500 / 256 / 0 | 7485 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | K | 480 / 256 / 0 | 480 / 256 / 0 | 6912 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 6988 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | K - 112 | 484 / 256 / 0 | 484 / 256 / 0 | 6987 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 6941 | 2 | 256 | FAIL |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=chunk | K | 472 / 256 / 0 | 472 / 256 / 0 | 7109 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=chunk | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 7203 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=chunk | K - 112 | 488 / 256 / 0 | 488 / 256 / 0 | 7181 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=chunk | K - 112, all edges | 496 / 256 / 0 | 496 / 256 / 0 | 7114 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=chunk | K | 472 / 256 / 0 | 472 / 256 / 0 | 6848 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=chunk | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 6924 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=chunk | K - 112 | 484 / 256 / 0 | 484 / 256 / 0 | 6923 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=chunk | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 6874 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | K | 472 / 256 / 0 | 472 / 256 / 0 | 4932 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 4992 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | K - 112 | 484 / 256 / 0 | 484 / 256 / 0 | 5008 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 4845 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | K | 468 / 256 / 0 | 468 / 256 / 0 | 4944 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 5026 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | K - 112 | 480 / 256 / 0 | 480 / 256 / 0 | 5021 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 4870 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=tile | K | 472 / 256 / 0 | 472 / 256 / 0 | 5315 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=tile | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 5373 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=tile | K - 112 | 484 / 256 / 0 | 484 / 256 / 0 | 5389 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=tile | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 5224 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=tile | K | 468 / 256 / 0 | 468 / 256 / 0 | 5384 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=tile | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 5455 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=tile | K - 112 | 480 / 256 / 0 | 480 / 256 / 0 | 5451 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=tile | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 5296 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | K | 472 / 256 / 0 | 472 / 256 / 0 | 4950 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 5010 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | K - 112 | 484 / 256 / 0 | 484 / 256 / 0 | 5026 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 4858 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | K | 468 / 256 / 0 | 468 / 256 / 0 | 4977 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 5058 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | K - 112 | 480 / 256 / 0 | 480 / 256 / 0 | 5053 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 4902 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=chunk | K | 472 / 256 / 0 | 472 / 256 / 0 | 5333 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=chunk | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 5391 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=chunk | K - 112 | 484 / 256 / 0 | 484 / 256 / 0 | 5407 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=chunk | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 5242 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=chunk | K | 468 / 256 / 0 | 468 / 256 / 0 | 5417 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=chunk | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 5486 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=chunk | K - 112 | 480 / 256 / 0 | 480 / 256 / 0 | 5482 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=chunk | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 5328 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | K | 472 / 256 / 0 | 472 / 256 / 0 | 5293 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | K - 16 | 476 / 256 / 0 | 476 / 256 / 0 | 5375 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | K - 112 | 484 / 256 / 0 | 484 / 256 / 0 | 5358 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 5393 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | K | 472 / 256 / 0 | 472 / 256 / 0 | 5299 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | K - 16 | 476 / 256 / 0 | 476 / 256 / 0 | 5382 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | K - 112 | 484 / 256 / 0 | 484 / 256 / 0 | 5365 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 5395 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | K | 472 / 256 / 0 | 472 / 256 / 0 | 5312 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | K - 16 | 476 / 256 / 0 | 476 / 256 / 0 | 5394 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | K - 112 | 484 / 256 / 0 | 484 / 256 / 0 | 5377 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 5411 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | K | 472 / 256 / 0 | 472 / 256 / 0 | 5329 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | K - 16 | 476 / 256 / 0 | 476 / 256 / 0 | 5412 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | K - 112 | 484 / 256 / 0 | 484 / 256 / 0 | 5395 | 0 | 256 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 5430 | 0 | 256 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, EXCHANGE=l2 | K | 468 / 256 / 0 | 468 / 256 / 0 | 4859 | 0 | 256 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, EXCHANGE=l2 | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 4932 | 0 | 256 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, EXCHANGE=l2 | K - 112 | 480 / 256 / 0 | 480 / 256 / 0 | 4927 | 0 | 256 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, EXCHANGE=l2 | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 4779 | 0 | 256 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=64, PARTIALS=fp32, EXCHANGE=l2 | K | 468 / 256 / 0 | 468 / 256 / 0 | 5038 | 0 | 256 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=64, PARTIALS=fp32, EXCHANGE=l2 | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 5112 | 0 | 256 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=64, PARTIALS=fp32, EXCHANGE=l2 | K - 112 | 480 / 256 / 0 | 480 / 256 / 0 | 5107 | 0 | 256 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=64, PARTIALS=fp32, EXCHANGE=l2 | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 4963 | 0 | 256 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, EXCHANGE=l2 | K | 468 / 256 / 0 | 468 / 256 / 0 | 5299 | 0 | 256 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, EXCHANGE=l2 | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 5361 | 0 | 256 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, EXCHANGE=l2 | K - 112 | 480 / 256 / 0 | 480 / 256 / 0 | 5357 | 0 | 256 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, EXCHANGE=l2 | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 5203 | 0 | 256 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=64, PARTIALS=bf16_pairs, EXCHANGE=l2 | K | 468 / 256 / 0 | 468 / 256 / 0 | 5303 | 0 | 256 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=64, PARTIALS=bf16_pairs, EXCHANGE=l2 | K - 16 | 480 / 256 / 0 | 480 / 256 / 0 | 5376 | 0 | 256 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=64, PARTIALS=bf16_pairs, EXCHANGE=l2 | K - 112 | 480 / 256 / 0 | 480 / 256 / 0 | 5371 | 0 | 256 | identical |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=64, PARTIALS=bf16_pairs, EXCHANGE=l2 | K - 112, all edges | 492 / 256 / 0 | 492 / 256 / 0 | 5231 | 0 | 256 | identical |

FAIL
