# v24 asm check

4096x8192xK, GEMM_TILE_LIMIT = 256 + S.

## Against v23 (same configuration, kernel name replaced, comments and directives dropped)

| K | S | configuration | v23 VGPRs / spills | v24 VGPRs / spills | TTGIR lines changed | instructions | instruction lines changed | result |
|---|---|---|---|---|---|---|---|---|
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128 | 448 / 0 | 448 / 0 | 0 | 4331 / 4331 | 0 | identical |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100 | 448 / 0 | 448 / 0 | 0 | 4324 / 4324 | 0 | identical |
| 8192 | 38 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=152 | 452 / 0 | 452 / 0 | 0 | 5557 / 5557 | 0 | identical |
| 4096 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs | 444 / 0 | 444 / 0 | 0 | 4782 / 4776 | 74 | same TTGIR, LLVM order differs |
| 4096 | 16 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=64, PARTIALS=bf16 | 452 / 0 | 452 / 0 | 0 | 6282 / 6282 | 0 | identical |
| 8192 | 24 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=144, RS_READERS=all | 452 / 0 | 452 / 0 | 0 | 5324 / 5324 | 0 | identical |
| 8192 | 16 | POLICY=two_tile | 512 / 15 | 512 / 15 | 0 | 7201 / 7201 | 0 | identical |
| 16384 | 192 | POLICY=spread | 512 / 15 | 512 / 15 | 0 | 7224 / 7224 | 0 | identical |
| 8192 | 16 | POLICY=dp | 440 / 0 | 440 / 0 | 0 | 2720 / 2720 | 0 | identical |
| 8192 | 16 | (defaults) | 448 / 0 | 448 / 0 | 0 | 4323 / 4323 | 0 | identical |

## v24's options

| K | S | configuration | VGPRs | spills | atomic adds | partial stores | partial and slice loads | checks |
|---|---|---|---|---|---|---|---|---|
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | 480 | 0 | - | `buffer_store_dwordx4 sc0 sc1` 64 | `buffer_load_dwordx4 sc0 sc1` 192 | ok |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | 480 | 0 | - | `buffer_store_dwordx4 sc0 sc1` 64 | `buffer_load_dwordx4 sc0 sc1` 192 | ok |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=tile | 452 | 0 | - | `buffer_store_dwordx4 sc0 sc1` 32 | `buffer_load_dwordx4 sc0 sc1` 32 | ok |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=tile | 452 | 0 | - | `buffer_store_dwordx4 sc0 sc1` 32 | `buffer_load_dwordx4 sc0 sc1` 32 | ok |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | 484 | 0 | - | `buffer_store_dwordx4 sc0 sc1` 64 | `buffer_load_dwordx4 sc0 sc1` 192 | ok |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | 480 | 0 | - | `buffer_store_dwordx4 sc0 sc1` 64 | `buffer_load_dwordx4 sc0 sc1` 192 | ok |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=chunk | 452 | 0 | - | `buffer_store_dwordx4 sc0 sc1` 32 | `buffer_load_dwordx4 sc0 sc1` 32 | ok |
| 8192 | 8 | POLICY=one_tile, FIXUP=owner, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=chunk | 452 | 0 | - | `buffer_store_dwordx4 sc0 sc1` 32 | `buffer_load_dwordx4 sc0 sc1` 32 | ok |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | 448 | 0 | - | `buffer_store_dwordx4 sc0 sc1` 64 | `buffer_load_dwordx4 sc0 sc1` 16 | ok |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | 448 | 0 | - | `buffer_store_dwordx4 sc0 sc1` 64 | `buffer_load_dwordx4 sc0 sc1` 16 | ok |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=tile | 448 | 0 | - | `buffer_store_dwordx4 sc0 sc1` 32 | `buffer_load_dwordx4 sc0 sc1` 8 | ok |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=tile | 444 | 0 | - | `buffer_store_dwordx4 sc0 sc1` 32 | `buffer_load_dwordx4 sc0 sc1` 8 | ok |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | 448 | 0 | - | `buffer_store_dwordx4 sc0 sc1` 64 | `buffer_load_dwordx4 sc0 sc1` 16 | ok |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | 448 | 0 | - | `buffer_store_dwordx4 sc0 sc1` 64 | `buffer_load_dwordx4 sc0 sc1` 16 | ok |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, PLACEMENT=chunk | 448 | 0 | - | `buffer_store_dwordx4 sc0 sc1` 32 | `buffer_load_dwordx4 sc0 sc1` 8 | ok |
| 8192 | 8 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=100, PARTIALS=bf16_pairs, PLACEMENT=chunk | 444 | 0 | - | `buffer_store_dwordx4 sc0 sc1` 32 | `buffer_load_dwordx4 sc0 sc1` 8 | ok |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=tile | 448 | 0 | `buffer_atomic_pk_add_bf16` 128 | `buffer_store_dwordx2 sc0 sc1` 2 | - | ok |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=tile | 448 | 0 | `buffer_atomic_pk_add_bf16` 128 | `buffer_store_dwordx2 sc0 sc1` 2 | - | ok |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=128, PARTIALS=fp32, PLACEMENT=chunk | 448 | 0 | `buffer_atomic_pk_add_bf16` 128 | `buffer_store_dwordx2 sc0 sc1` 2 | - | ok |
| 8192 | 8 | POLICY=one_tile, FIXUP=atomic_iz, NUM_PROGRAMS=100, PARTIALS=fp32, PLACEMENT=chunk | 448 | 0 | `buffer_atomic_pk_add_bf16` 128 | `buffer_store_dwordx2 sc0 sc1` 2 | - | ok |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=fp32, EXCHANGE=l2 | 448 | 0 | - | `buffer_store_dwordx4` 100 | `buffer_load_dword sc0 nt` 1, `buffer_load_dwordx4 sc0 nt` 16 | ok |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=64, PARTIALS=fp32, EXCHANGE=l2 | 448 | 0 | - | `buffer_store_dwordx4` 104 | `buffer_load_dword sc0 nt` 1, `buffer_load_dwordx4 sc0 nt` 64 | ok |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=128, PARTIALS=bf16_pairs, EXCHANGE=l2 | 444 | 0 | - | `buffer_store_dwordx4` 68 | `buffer_load_dword sc0 nt` 1, `buffer_load_dwordx4 sc0 nt` 8 | ok |
| 8192 | 16 | POLICY=one_tile, FIXUP=rs, NUM_PROGRAMS=64, PARTIALS=bf16_pairs, EXCHANGE=l2 | 448 | 0 | - | `buffer_store_dwordx4` 72 | `buffer_load_dword sc0 nt` 1, `buffer_load_dwordx4 sc0 nt` 8 | ok |

PASS
