4352x4096x8192, one-tile reduce-scatter (capped rule). Whole-kernel counts of vector memory instructions by cache bits.

## mem: 484 VGPRs, 0 spills

- `buffer_inv sc1`: 2
- `buffer_load_dwordx4`: 192
- `buffer_load_dwordx4 sc0 sc1`: 16
- `buffer_store_dwordx2`: 8
- `buffer_store_dwordx4`: 32
- `buffer_store_dwordx4 sc0 sc1`: 64
- `buffer_wbl2 sc1`: 2
- `global_atomic_add sc0`: 1
- `global_atomic_swap`: 2
- `global_load_dword sc1`: 3

- check write-through partial stores: ok
- check buffer_wbl2 present: ok
- check buffer_inv sc1 present: ok

## wt: 484 VGPRs, 0 spills

- `buffer_load_dwordx4`: 192
- `buffer_load_dwordx4 sc0 sc1`: 16
- `buffer_store_dwordx2`: 8
- `buffer_store_dwordx4`: 32
- `buffer_store_dwordx4 sc0 sc1`: 64
- `global_atomic_add sc0`: 1
- `global_atomic_swap`: 2
- `global_load_dword sc1`: 2

- check write-through partial stores: ok
- check no buffer_wbl2: ok
- check no buffer_inv sc1: ok

## l2: 484 VGPRs, 0 spills

- `buffer_load_dword sc0 nt`: 1
- `buffer_load_dwordx4`: 192
- `buffer_load_dwordx4 sc0 nt`: 16
- `buffer_store_dwordx2`: 8
- `buffer_store_dwordx4`: 96
- `global_store_dword`: 1

- check no write-through store: ok
- check no sc0 sc1 load: ok
- check no buffer_wbl2: ok
- check no buffer_inv sc1: ok
- check no buffer_inv: ok
- check sc0 nt slice loads: ok

