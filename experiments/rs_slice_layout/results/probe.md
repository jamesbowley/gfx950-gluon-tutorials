HIP_VISIBLE_DEVICES=1, AMD Instinct MI355X, 2026-10-08 11:30

## probe

One 128x128 quadrant of bf16 values in a source layout, converted REPS = 1000 times in a loop on one CU (4 warps); 'ns per conversion' is the loop time per iteration (median of 20 launches) minus the loop with no conversion in the same source layout. Instruction counts are the loop body's; 'LDS rounds' is its s_barrier count / 2. 'checked': the converted values stored row-major equal every (row, col). Each loop iteration also does an add, a bf16 convert and a select over the whole quadrant to keep every conversion live; that is the no-conversion baseline.

| case | conversion | trivial | instructions in the loop | LDS rounds | LDS bytes | ns per iteration | ns per conversion | VGPRs | spills | checked |
|---|---|---|---|---|---|---|---|---|---|---|
| mfma_add | MFMA, no conversion (loop baseline) | yes | 32 v_perm_b32 | 0 | 0 | 1890.8 | 0.0 | 296 | 0 | ok |
| mfma_to_store | MFMA -> gStoreLayoutC (v20's owner / data-parallel C store) | no | 8 ds_read_b128, 8 ds_write_b128, 32 v_perm_b32 | 2 | 16384 | 2259.3 | 368.5 | 296 | 0 | ok |
| l3_add | L3, no conversion (loop baseline) | yes | 32 v_perm_b32 | 0 | 0 | 1896.8 | 0.0 | 296 | 0 | ok |
| l3_to_s3 | L3 -> 3-D store layout, 8 bf16 per lane (a coalesced slice store) | no | 8 ds_read_b128, 8 ds_write_b128, 32 v_perm_b32 | 4 | 8192 | 2407.3 | 510.4 | 294 | 0 | ok |
| l3_to_l3old | L3 -> columns-first L3 (v21's dropped attempt) | no | 8 ds_read_b128, 8 ds_write_b128, 32 v_perm_b32 | 8 | 4096 | 2544.3 | 647.5 | 294 | 0 | ok |
| l3_to_mfma | L3 reshaped to 128x128 -> MFMA (warp order differs) | no | 8 ds_read_b128, 8 ds_write_b128, 32 v_perm_b32 | 8 | 4096 | 2562.1 | 665.3 | 294 | 0 | ok |
| l3_to_mfma_to_store | L3 reshaped -> MFMA -> gStoreLayoutC | no | 16 ds_read_b128, 16 ds_write_b128, 32 v_perm_b32 | 10 | 16384 | 2809.9 | 913.0 | 188 | 0 | ok |
| l3m_to_mfma | L3M reshaped -> MFMA, assert_trivial=True | yes | 32 v_perm_b32 | 0 | 0 | 1896.3 | 0.0 | 298 | 0 | ok |
| l3m_to_mfma_to_store | L3M reshaped -> MFMA (trivial) -> gStoreLayoutC | no | 8 ds_read_b128, 8 ds_write_b128, 32 v_perm_b32 | 2 | 16384 | 2259.2 | 362.4 | 296 | 0 | ok |
| l3_to_store2d | L3 reshaped -> gStoreLayoutC directly | no | 8 ds_read_b128, 8 ds_write_b128, 32 v_perm_b32 | 2 | 16384 | 2262.4 | 365.6 | 294 | 0 | ok |
| l3_to_s3row | L3 -> gStoreLayoutC's mapping written as a 3-D layout (no reshape) | no | 8 ds_read_b128, 8 ds_write_b128, 32 v_perm_b32 | 2 | 16384 | 2278.2 | 381.4 | 296 | 0 | ok |
