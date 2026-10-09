# v23 asm check

4096x8192x16384, GEMM_TILE_LIMIT = 256 + S, STREAMK_NUM_PROGRAMS = S n, STREAMK_RS_READERS=pow2.

## Reduce-scatter slice (tile barrier's last fence to s_endpgm)

| version | S | n | NB | partials | VGPRs | spills | slice instructions | checks |
|---|---|---|---|---|---|---|---|---|
| v20 | 4 | 4 | 16 | fp32 | 448 | 0 | `buffer_load_dwordx4 sc0 sc1` 16, `buffer_store_dwordx2` 16 | ok |
| v23 | 4 | 4 | 16 | fp32 | 448 | 0 | `buffer_load_dwordx4 sc0 sc1` 16, `buffer_store_dwordx4` 8, `ds_read_b128` 8, `ds_write_b128` 8, `s_barrier` 4 | ok |
| v23 | 4 | 4 | 16 | bf16 | 448 | 0 | `buffer_load_dwordx2 sc0 sc1` 16, `buffer_store_dwordx4` 8, `ds_read_b128` 8, `ds_write_b128` 8, `s_barrier` 4 | ok |
| v23 | 4 | 4 | 16 | bf16_pairs | 448 | 0 | `buffer_load_dwordx4 sc0 sc1` 8, `buffer_store_dwordx4` 8, `ds_read_b128` 8, `ds_write_b128` 8, `s_barrier` 4 | ok |
| v20 | 8 | 8 | 8 | fp32 | 448 | 0 | `buffer_load_dwordx4 sc0 sc1` 16, `buffer_store_dwordx2` 8 | ok |
| v23 | 8 | 8 | 8 | fp32 | 448 | 0 | `buffer_load_dwordx4 sc0 sc1` 16, `buffer_store_dwordx4` 4, `ds_read_b128` 4, `ds_write_b128` 4, `s_barrier` 2 | ok |
| v23 | 8 | 8 | 8 | bf16 | 448 | 0 | `buffer_load_dwordx2 sc0 sc1` 16, `buffer_store_dwordx4` 4, `ds_read_b128` 4, `ds_write_b128` 4, `s_barrier` 2 | ok |
| v23 | 8 | 8 | 8 | bf16_pairs | 444 | 0 | `buffer_load_dwordx4 sc0 sc1` 8, `buffer_store_dwordx4` 4, `ds_read_b128` 4, `ds_write_b128` 4, `s_barrier` 2 | ok |
| v20 | 16 | 16 | 4 | fp32 | 448 | 0 | `buffer_load_dwordx4 sc0 sc1` 16, `buffer_store_dwordx2` 4 | ok |
| v23 | 16 | 16 | 4 | fp32 | 448 | 0 | `buffer_load_dwordx4 sc0 sc1` 16, `buffer_store_dwordx4` 2, `ds_read2_b64` 2, `ds_write_b64` 4, `s_barrier` 2 | ok |
| v23 | 16 | 16 | 4 | bf16 | 448 | 0 | `buffer_load_dwordx2 sc0 sc1` 16, `buffer_store_dwordx4` 2, `ds_read2_b64` 2, `ds_write_b64` 4, `s_barrier` 2 | ok |
| v23 | 16 | 16 | 4 | bf16_pairs | 448 | 0 | `buffer_load_dwordx4 sc0 sc1` 8, `buffer_store_dwordx4` 2, `ds_read2_b64` 2, `ds_write_b64` 4, `s_barrier` 2 | ok |
| v20 | 4 | 32 | 2 | fp32 | 448 | 0 | `buffer_load_dwordx4 sc0 sc1` 16, `buffer_store_dwordx2` 2 | ok |
| v23 | 4 | 32 | 2 | fp32 | 448 | 0 | `buffer_load_dwordx4 sc0 sc1` 16, `buffer_store_dwordx4` 1, `ds_read_b64` 2, `ds_write_b64` 2, `s_barrier` 2 | ok |
| v23 | 4 | 32 | 2 | bf16 | 448 | 0 | `buffer_load_dwordx2 sc0 sc1` 16, `buffer_store_dwordx4` 1, `ds_read_b64` 2, `ds_write_b64` 2, `s_barrier` 2 | ok |
| v23 | 4 | 32 | 2 | bf16_pairs | 448 | 0 | `buffer_load_dwordx2 sc0 sc1` 16, `buffer_store_dwordx4` 1, `ds_read_b64` 2, `ds_write_b64` 2, `s_barrier` 2 | ok |
| v20 | 2 | 64 | 1 | fp32 | 448 | 0 | `buffer_load_dwordx4 sc0 sc1` 16, `buffer_store_dwordx2` 1 | ok |
| v23 | 2 | 64 | 1 | fp32 | 452 | 0 | `buffer_load_dwordx4 sc0 sc1` 16, `buffer_store_dwordx2` 1, `ds_read_b64` 1, `ds_write_b64` 1, `s_barrier` 2 | ok |
| v23 | 2 | 64 | 1 | bf16 | 448 | 0 | `buffer_load_dwordx2 sc0 sc1` 16, `buffer_store_dwordx2` 1, `ds_read_b64` 1, `ds_write_b64` 1, `s_barrier` 2 | ok |
| v23 | 2 | 64 | 1 | bf16_pairs | 448 | 0 | `buffer_load_dwordx2 sc0 sc1` 16, `buffer_store_dwordx2` 1, `ds_read_b64` 1, `ds_write_b64` 1, `s_barrier` 2 | ok |
| v20 | 16 | 12 | 8 | fp32 | 448 | 0 | `buffer_load_dwordx4 sc0 sc1` 16, `buffer_store_dwordx2` 8 | ok |
| v23 | 16 | 12 | 8 | fp32 | 448 | 0 | `buffer_load_dwordx4 sc0 sc1` 16, `buffer_store_dwordx4` 4, `ds_read_b128` 4, `ds_write_b128` 4, `s_barrier` 2 | ok |
| v23 | 16 | 12 | 8 | bf16 | 448 | 0 | `buffer_load_dwordx2 sc0 sc1` 16, `buffer_store_dwordx4` 4, `ds_read_b128` 4, `ds_write_b128` 4, `s_barrier` 2 | ok |
| v23 | 16 | 12 | 8 | bf16_pairs | 444 | 0 | `buffer_load_dwordx4 sc0 sc1` 8, `buffer_store_dwordx4` 4, `ds_read_b128` 4, `ds_write_b128` 4, `s_barrier` 2 | ok |

## Partial stores and owner reads (whole kernel, write-through stores and .cv loads)

| version | ending | S | n | partials | VGPRs | spills | partial stores (sc0 sc1) | partial loads (sc0 sc1) | checks |
|---|---|---|---|---|---|---|---|---|---|
| v20 | owner | 32 | 4 | fp32 | 480 | 0 | `buffer_store_dwordx4` 64 | `buffer_load_dwordx4` 192 | ok |
| v23 | owner | 32 | 4 | fp32 | 480 | 0 | `buffer_store_dwordx4` 64 | `buffer_load_dwordx4` 192 | ok |
| v23 | owner | 32 | 4 | bf16 | 452 | 0 | `buffer_store_dwordx2` 64 | `buffer_load_dwordx2` 64 | ok |
| v23 | owner | 32 | 4 | bf16_pairs | 452 | 0 | `buffer_store_dwordx4` 32 | `buffer_load_dwordx4` 32 | ok |
| v20 | rs | 16 | 16 | fp32 | 448 | 0 | `buffer_store_dwordx4` 64 | `buffer_load_dwordx4` 16 | ok |
| v23 | rs | 16 | 16 | fp32 | 448 | 0 | `buffer_store_dwordx4` 64 | `buffer_load_dwordx4` 16 | ok |
| v23 | rs | 16 | 16 | bf16 | 448 | 0 | `buffer_store_dwordx2` 64 | `buffer_load_dwordx2` 16 | ok |
| v23 | rs | 16 | 16 | bf16_pairs | 448 | 0 | `buffer_store_dwordx4` 32 | `buffer_load_dwordx4` 8 | ok |

PASS
