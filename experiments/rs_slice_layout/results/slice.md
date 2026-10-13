HIP_VISIBLE_DEVICES=1, AMD Instinct MI355X, 2026-10-08 09:52

## slice

Synthetic one-tile reduce-scatter (slice_kernel.py): W programs, n per 256x256 tile, every tile on one XCD; `.wt` fp32 partials, drain, relaxed gpu counter, `.cv` loads, bf16 C. Variants: v20 (L3 slice, 8 B per lane C store), v20_old_cvt (L3, then convert to columns-first L3: v21's dropped attempt), v20_store2d (L3, reshaped to the slice's band and converted to gStoreLayoutC), mfma_read (slice loaded in the MFMA layout, owner-style convert to gStoreLayoutC), writer_cvt (the writer converts to 4 fp32 per lane and stores row-major; the reader loads 8 per lane and stores C with no conversion), writer_cvt8 (the writer converts to gStoreLayoutC itself: 32 bytes per lane, so each `.wt` store instruction leaves 16-byte holes). Medians over 30 cold launches (1 GiB read before each) and over groups of each group's critical path; total is the makespan. Every C element checked.

| variant | setup | n | W | total us | publish us | sync us | read us | C store us | wrong |
|---|---|---|---|---|---|---|---|---|---|
| v20 | tiles 16 | 2 | 32 | 15.54 | 6.71 | 0.60 | 2.80 | 4.48 | 0 |
| v20_old_cvt | tiles 16 | 2 | 32 | 15.42 | 6.44 | 0.64 | 3.84 | 2.64 | 0 |
| v20_store2d | tiles 16 | 2 | 32 | 12.88 | 6.66 | 0.56 | 2.68 | 1.68 | 0 |
| mfma_read | tiles 16 | 2 | 32 | 12.79 | 6.42 | 0.56 | 2.64 | 1.72 | 0 |
| writer_cvt | tiles 16 | 2 | 32 | 17.69 | 10.82 | 0.64 | 4.48 | 0.80 | 0 |
| writer_cvt8 | tiles 16 | 2 | 32 | 24.27 | 17.04 | 0.60 | 4.40 | 0.80 | 0 |
| v20 | tiles 16 | 4 | 64 | 16.68 | 7.20 | 0.60 | 5.44 | 2.20 | 0 |
| v20_old_cvt | tiles 16 | 4 | 64 | 14.62 | 7.16 | 0.64 | 3.12 | 1.80 | 0 |
| v20_store2d | tiles 16 | 4 | 64 | 13.02 | 7.44 | 0.60 | 3.08 | 1.08 | 0 |
| mfma_read | tiles 16 | 4 | 64 | 12.94 | 7.08 | 0.56 | 3.00 | 0.96 | 0 |
| writer_cvt | tiles 16 | 4 | 64 | 18.68 | 12.01 | 0.60 | 4.96 | 0.52 | 0 |
| writer_cvt8 | tiles 16 | 4 | 64 | 26.67 | 19.72 | 0.60 | 4.86 | 0.48 | 0 |
| v20 | tiles 16 | 8 | 128 | 15.60 | 7.73 | 0.68 | 4.84 | 1.16 | 0 |
| v20_old_cvt | tiles 16 | 8 | 128 | 16.56 | 7.25 | 0.60 | 4.68 | 1.12 | 0 |
| v20_store2d | tiles 16 | 8 | 128 | 16.56 | 8.21 | 0.64 | 4.68 | 0.68 | 0 |
| mfma_read | tiles 16 | 8 | 128 | 16.90 | 8.72 | 0.72 | 4.84 | 0.60 | 0 |
| writer_cvt | tiles 16 | 8 | 128 | 19.80 | 12.42 | 0.80 | 5.40 | 0.40 | 0 |
| writer_cvt8 | tiles 16 | 8 | 128 | 32.12 | 24.93 | 0.72 | 5.32 | 0.40 | 0 |
| v20 | tiles 16 | 16 | 256 | 24.90 | 11.52 | 0.72 | 9.96 | 0.72 | 0 |
| v20_old_cvt | tiles 16 | 16 | 256 | 24.81 | 10.90 | 1.04 | 10.12 | 0.56 | 0 |
| v20_store2d | tiles 16 | 16 | 256 | 24.58 | 10.93 | 1.08 | 10.20 | 0.44 | 0 |
| mfma_read | tiles 16 | 16 | 256 | 24.46 | 9.80 | 1.16 | 10.04 | 0.44 | 0 |
| writer_cvt | tiles 16 | 16 | 256 | 31.08 | 15.68 | 0.92 | 12.20 | 0.40 | 0 |
| writer_cvt8 | tiles 16 | 16 | 256 | 57.06 | 43.66 | 0.94 | 8.48 | 0.36 | 0 |
| v20 | CUs 256 | 2 | 256 | 26.73 | 12.36 | 0.72 | 8.00 | 4.48 | 0 |
| v20_old_cvt | CUs 256 | 2 | 256 | 26.84 | 12.93 | 0.76 | 8.92 | 3.52 | 0 |
| v20_store2d | CUs 256 | 2 | 256 | 25.04 | 13.09 | 0.64 | 8.32 | 2.28 | 0 |
| mfma_read | CUs 256 | 2 | 256 | 23.84 | 11.28 | 0.64 | 8.24 | 2.28 | 0 |
| writer_cvt | CUs 256 | 2 | 256 | 26.94 | 16.00 | 0.72 | 8.44 | 1.36 | 0 |
| writer_cvt8 | CUs 256 | 2 | 256 | 55.66 | 40.62 | 0.76 | 7.88 | 1.00 | 0 |
| v20 | CUs 256 | 4 | 256 | 26.40 | 11.88 | 0.68 | 8.64 | 2.16 | 0 |
| v20_old_cvt | CUs 256 | 4 | 256 | 24.81 | 12.06 | 0.68 | 8.08 | 2.16 | 0 |
| v20_store2d | CUs 256 | 4 | 256 | 22.64 | 11.76 | 0.68 | 8.28 | 1.16 | 0 |
| mfma_read | CUs 256 | 4 | 256 | 24.74 | 13.08 | 0.64 | 8.16 | 1.12 | 0 |
| writer_cvt | CUs 256 | 4 | 256 | 27.06 | 16.56 | 0.64 | 8.72 | 0.68 | 0 |
| writer_cvt8 | CUs 256 | 4 | 256 | 55.45 | 41.48 | 0.64 | 7.54 | 0.56 | 0 |
| v20 | CUs 256 | 8 | 256 | 22.96 | 10.93 | 0.80 | 9.28 | 1.20 | 0 |
| v20_old_cvt | CUs 256 | 8 | 256 | 25.18 | 10.00 | 0.80 | 9.32 | 1.12 | 0 |
| v20_store2d | CUs 256 | 8 | 256 | 24.98 | 11.84 | 0.84 | 8.36 | 0.64 | 0 |
| mfma_read | CUs 256 | 8 | 256 | 23.74 | 11.89 | 0.92 | 8.56 | 0.60 | 0 |
| writer_cvt | CUs 256 | 8 | 256 | 26.43 | 15.87 | 0.80 | 8.64 | 0.44 | 0 |
| writer_cvt8 | CUs 256 | 8 | 256 | 55.36 | 41.66 | 0.76 | 7.72 | 0.40 | 0 |
| v20 | CUs 256 | 16 | 256 | 24.78 | 11.40 | 0.72 | 9.92 | 0.72 | 0 |
| v20_old_cvt | CUs 256 | 16 | 256 | 24.72 | 10.92 | 1.04 | 10.20 | 0.56 | 0 |
| v20_store2d | CUs 256 | 16 | 256 | 24.55 | 10.92 | 1.08 | 10.16 | 0.44 | 0 |
| mfma_read | CUs 256 | 16 | 256 | 24.33 | 9.84 | 1.20 | 10.04 | 0.44 | 0 |
| writer_cvt | CUs 256 | 16 | 256 | 31.09 | 15.68 | 0.92 | 12.14 | 0.40 | 0 |
| writer_cvt8 | CUs 256 | 16 | 256 | 57.32 | 43.41 | 0.88 | 8.40 | 0.36 | 0 |

### Compiled code

Whole kernel, per variant and n: memory and LDS instructions by width, s_barrier count (including the stamps' and the drain's), VGPRs and spills.

| variant | n | instructions | s_barrier | VGPRs | spills |
|---|---|---|---|---|---|
| mfma_read | 2 | 64 buffer_load_dwordx4, 80 buffer_store_dwordx4, 16 ds_read_b128, 2 ds_read_b32, 16 ds_write_b128, 2 ds_write_b32, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 19 | 260 | 0 |
| v20 | 2 | 64 buffer_load_dwordx4, 32 buffer_store_dwordx2, 64 buffer_store_dwordx4, 2 ds_read_b32, 2 ds_write_b32, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 11 | 264 | 0 |
| v20_old_cvt | 2 | 64 buffer_load_dwordx4, 32 buffer_store_dwordx2, 64 buffer_store_dwordx4, 16 ds_read_b128, 2 ds_read_b32, 16 ds_write_b128, 2 ds_write_b32, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 43 | 239 | 0 |
| v20_store2d | 2 | 64 buffer_load_dwordx4, 80 buffer_store_dwordx4, 16 ds_read_b128, 2 ds_read_b32, 16 ds_write_b128, 2 ds_write_b32, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 19 | 257 | 0 |
| writer_cvt | 2 | 64 buffer_load_dwordx4, 80 buffer_store_dwordx4, 64 ds_read_b128, 2 ds_read_b32, 64 ds_write_b128, 2 ds_write_b32, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 43 | 227 | 0 |
| writer_cvt8 | 2 | 64 buffer_load_dwordx4, 80 buffer_store_dwordx4, 64 ds_read_b128, 2 ds_read_b32, 64 ds_write_b128, 2 ds_write_b32, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 43 | 227 | 0 |
| mfma_read | 4 | 64 buffer_load_dwordx4, 72 buffer_store_dwordx4, 8 ds_read_b128, 2 ds_read_b32, 8 ds_write_b128, 2 ds_write_b32, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 15 | 127 | 0 |
| v20 | 4 | 64 buffer_load_dwordx4, 16 buffer_store_dwordx2, 64 buffer_store_dwordx4, 2 ds_read_b32, 2 ds_write_b32, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 11 | 94 | 0 |
| v20_old_cvt | 4 | 64 buffer_load_dwordx4, 16 buffer_store_dwordx2, 64 buffer_store_dwordx4, 8 ds_read_b128, 2 ds_read_b32, 8 ds_write_b128, 2 ds_write_b32, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 27 | 127 | 0 |
| v20_store2d | 4 | 64 buffer_load_dwordx4, 72 buffer_store_dwordx4, 8 ds_read_b128, 2 ds_read_b32, 8 ds_write_b128, 2 ds_write_b32, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 15 | 126 | 0 |
| writer_cvt | 4 | 64 buffer_load_dwordx4, 72 buffer_store_dwordx4, 64 ds_read_b128, 2 ds_read_b32, 64 ds_write_b128, 2 ds_write_b32, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 43 | 133 | 0 |
| writer_cvt8 | 4 | 64 buffer_load_dwordx4, 72 buffer_store_dwordx4, 64 ds_read_b128, 2 ds_read_b32, 64 ds_write_b128, 2 ds_write_b32, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 43 | 134 | 0 |
| mfma_read | 8 | 64 buffer_load_dwordx4, 68 buffer_store_dwordx4, 4 ds_read_b128, 2 ds_read_b32, 4 ds_write_b128, 2 ds_write_b32, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 13 | 101 | 0 |
| v20 | 8 | 64 buffer_load_dwordx4, 8 buffer_store_dwordx2, 64 buffer_store_dwordx4, 2 ds_read_b32, 2 ds_write_b32, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 11 | 83 | 0 |
| v20_old_cvt | 8 | 64 buffer_load_dwordx4, 8 buffer_store_dwordx2, 64 buffer_store_dwordx4, 4 ds_read_b128, 2 ds_read_b32, 4 ds_write_b128, 2 ds_write_b32, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 19 | 102 | 0 |
| v20_store2d | 8 | 64 buffer_load_dwordx4, 68 buffer_store_dwordx4, 4 ds_read_b128, 2 ds_read_b32, 4 ds_write_b128, 2 ds_write_b32, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 13 | 102 | 0 |
| writer_cvt | 8 | 64 buffer_load_dwordx4, 68 buffer_store_dwordx4, 64 ds_read_b128, 2 ds_read_b32, 64 ds_write_b128, 2 ds_write_b32, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 43 | 142 | 0 |
| writer_cvt8 | 8 | 64 buffer_load_dwordx4, 68 buffer_store_dwordx4, 64 ds_read_b128, 2 ds_read_b32, 64 ds_write_b128, 2 ds_write_b32, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 43 | 142 | 0 |
| mfma_read | 16 | 64 buffer_load_dwordx4, 66 buffer_store_dwordx4, 2 ds_read2_b64, 2 ds_read_b32, 2 ds_write_b32, 4 ds_write_b64, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 13 | 113 | 0 |
| v20 | 16 | 64 buffer_load_dwordx4, 4 buffer_store_dwordx2, 64 buffer_store_dwordx4, 2 ds_read_b32, 2 ds_write_b32, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 11 | 94 | 0 |
| v20_old_cvt | 16 | 64 buffer_load_dwordx4, 4 buffer_store_dwordx2, 64 buffer_store_dwordx4, 2 ds_read_b128, 2 ds_read_b32, 2 ds_write_b128, 2 ds_write_b32, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 15 | 110 | 0 |
| v20_store2d | 16 | 64 buffer_load_dwordx4, 66 buffer_store_dwordx4, 2 ds_read2_b64, 2 ds_read_b32, 2 ds_write_b32, 4 ds_write_b64, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 13 | 112 | 0 |
| writer_cvt | 16 | 64 buffer_load_dwordx4, 66 buffer_store_dwordx4, 64 ds_read_b128, 2 ds_read_b32, 64 ds_write_b128, 2 ds_write_b32, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 43 | 146 | 0 |
| writer_cvt8 | 16 | 64 buffer_load_dwordx4, 66 buffer_store_dwordx4, 64 ds_read_b128, 2 ds_read_b32, 64 ds_write_b128, 2 ds_write_b32, 2 global_load_dword, 1 global_store_dwordx2, 3 global_store_dwordx4 | 43 | 147 | 0 |
