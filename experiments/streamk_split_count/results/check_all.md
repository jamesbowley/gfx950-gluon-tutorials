# check_all.py: STREAMK_NUM_PROGRAMS in v15-v20 (HIP_VISIBLE_DEVICES=3, 10 rotating launches)

| version | STREAMK_NUM_PROGRAMS | other env | shape | policy / fixup / order / bias / launches / wrong results / flags left up |
|---|---|---|---|---|
| v15_streamk_onetile | 4 | | 3328x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v15_streamk_onetile | 4 | | 3328x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v15_streamk_onetile | 8 | | 3328x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v15_streamk_onetile | 8 | | 3328x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v15_streamk_onetile | default | | 3328x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v15_streamk_onetile | default | | 3328x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v15_streamk_onetile | 16 | | 4352x4096x8192 | v9 | 0 | 10 | 0 | 0 |
| v15_streamk_onetile | 16 | | 4352x4096x8192 | v9 | 1 | 10 | 0 | 0 |
| v15_streamk_onetile | 32 | | 4352x4096x8192 | v9 | 0 | 10 | 0 | 0 |
| v15_streamk_onetile | 32 | | 4352x4096x8192 | v9 | 1 | 10 | 0 | 0 |
| v15_streamk_onetile | default | | 4352x4096x8192 | v9 | 0 | 10 | 0 | 0 |
| v15_streamk_onetile | default | | 4352x4096x8192 | v9 | 1 | 10 | 0 | 0 |
| v15_streamk_onetile | 38 | | 3584x5376x8192 | v9 | 0 | 10 | 0 | 0 |
| v15_streamk_onetile | 38 | | 3584x5376x8192 | v9 | 1 | 10 | 0 | 0 |
| v15_streamk_onetile | 76 | | 3584x5376x8192 | v9 | 0 | 10 | 0 | 0 |
| v15_streamk_onetile | 76 | | 3584x5376x8192 | v9 | 1 | 10 | 0 | 0 |
| v15_streamk_onetile | default | | 3584x5376x8192 | v9 | 0 | 10 | 0 | 0 |
| v15_streamk_onetile | default | | 3584x5376x8192 | v9 | 1 | 10 | 0 | 0 |
| v15_streamk_onetile | 64 | | 4096x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v15_streamk_onetile | 64 | | 4096x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v15_streamk_onetile | 128 | | 4096x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v15_streamk_onetile | 128 | | 4096x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v15_streamk_onetile | default | | 4096x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v15_streamk_onetile | default | | 4096x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v16_streamk_lane_partials | 4 | | 3328x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v16_streamk_lane_partials | 4 | | 3328x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v16_streamk_lane_partials | 8 | | 3328x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v16_streamk_lane_partials | 8 | | 3328x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v16_streamk_lane_partials | default | | 3328x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v16_streamk_lane_partials | default | | 3328x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v16_streamk_lane_partials | 16 | | 4352x4096x8192 | v9 | 0 | 10 | 0 | 0 |
| v16_streamk_lane_partials | 16 | | 4352x4096x8192 | v9 | 1 | 10 | 0 | 0 |
| v16_streamk_lane_partials | 32 | | 4352x4096x8192 | v9 | 0 | 10 | 0 | 0 |
| v16_streamk_lane_partials | 32 | | 4352x4096x8192 | v9 | 1 | 10 | 0 | 0 |
| v16_streamk_lane_partials | default | | 4352x4096x8192 | v9 | 0 | 10 | 0 | 0 |
| v16_streamk_lane_partials | default | | 4352x4096x8192 | v9 | 1 | 10 | 0 | 0 |
| v16_streamk_lane_partials | 38 | | 3584x5376x8192 | v9 | 0 | 10 | 0 | 0 |
| v16_streamk_lane_partials | 38 | | 3584x5376x8192 | v9 | 1 | 10 | 0 | 0 |
| v16_streamk_lane_partials | 76 | | 3584x5376x8192 | v9 | 0 | 10 | 0 | 0 |
| v16_streamk_lane_partials | 76 | | 3584x5376x8192 | v9 | 1 | 10 | 0 | 0 |
| v16_streamk_lane_partials | default | | 3584x5376x8192 | v9 | 0 | 10 | 0 | 0 |
| v16_streamk_lane_partials | default | | 3584x5376x8192 | v9 | 1 | 10 | 0 | 0 |
| v16_streamk_lane_partials | 64 | | 4096x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v16_streamk_lane_partials | 64 | | 4096x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v16_streamk_lane_partials | 128 | | 4096x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v16_streamk_lane_partials | 128 | | 4096x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v16_streamk_lane_partials | default | | 4096x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v16_streamk_lane_partials | default | | 4096x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | 4 | | 3328x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | 4 | | 3328x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | 8 | | 3328x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | 8 | | 3328x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | default | | 3328x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | default | | 3328x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | 16 | | 4352x4096x8192 | v9 | 0 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | 16 | | 4352x4096x8192 | v9 | 1 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | 32 | | 4352x4096x8192 | v9 | 0 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | 32 | | 4352x4096x8192 | v9 | 1 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | default | | 4352x4096x8192 | v9 | 0 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | default | | 4352x4096x8192 | v9 | 1 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | 38 | | 3584x5376x8192 | v9 | 0 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | 38 | | 3584x5376x8192 | v9 | 1 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | 76 | | 3584x5376x8192 | v9 | 0 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | 76 | | 3584x5376x8192 | v9 | 1 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | default | | 3584x5376x8192 | v9 | 0 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | default | | 3584x5376x8192 | v9 | 1 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | 64 | | 4096x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | 64 | | 4096x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | 128 | | 4096x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | 128 | | 4096x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | default | | 4096x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v17_streamk_tile_aligned | default | | 4096x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v18_streamk_chunk_major | 4 | | 3328x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v18_streamk_chunk_major | 4 | | 3328x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v18_streamk_chunk_major | 8 | | 3328x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v18_streamk_chunk_major | 8 | | 3328x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v18_streamk_chunk_major | default | | 3328x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v18_streamk_chunk_major | default | | 3328x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v18_streamk_chunk_major | 16 | | 4352x4096x8192 | v9 | 0 | 10 | 0 | 0 |
| v18_streamk_chunk_major | 16 | | 4352x4096x8192 | v9 | 1 | 10 | 0 | 0 |
| v18_streamk_chunk_major | 32 | | 4352x4096x8192 | v9 | 0 | 10 | 0 | 0 |
| v18_streamk_chunk_major | 32 | | 4352x4096x8192 | v9 | 1 | 10 | 0 | 0 |
| v18_streamk_chunk_major | default | | 4352x4096x8192 | v9 | 0 | 10 | 0 | 0 |
| v18_streamk_chunk_major | default | | 4352x4096x8192 | v9 | 1 | 10 | 0 | 0 |
| v18_streamk_chunk_major | 38 | | 3584x5376x8192 | v9 | 0 | 10 | 0 | 0 |
| v18_streamk_chunk_major | 38 | | 3584x5376x8192 | v9 | 1 | 10 | 0 | 0 |
| v18_streamk_chunk_major | 76 | | 3584x5376x8192 | v9 | 0 | 10 | 0 | 0 |
| v18_streamk_chunk_major | 76 | | 3584x5376x8192 | v9 | 1 | 10 | 0 | 0 |
| v18_streamk_chunk_major | default | | 3584x5376x8192 | v9 | 0 | 10 | 0 | 0 |
| v18_streamk_chunk_major | default | | 3584x5376x8192 | v9 | 1 | 10 | 0 | 0 |
| v18_streamk_chunk_major | 64 | | 4096x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v18_streamk_chunk_major | 64 | | 4096x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v18_streamk_chunk_major | 128 | | 4096x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v18_streamk_chunk_major | 128 | | 4096x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v18_streamk_chunk_major | default | | 4096x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v18_streamk_chunk_major | default | | 4096x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | 4 | | 3328x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | 4 | | 3328x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | 8 | | 3328x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | 8 | | 3328x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | default | | 3328x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | default | | 3328x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | 16 | | 4352x4096x8192 | v9 | 0 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | 16 | | 4352x4096x8192 | v9 | 1 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | 32 | | 4352x4096x8192 | v9 | 0 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | 32 | | 4352x4096x8192 | v9 | 1 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | default | | 4352x4096x8192 | v9 | 0 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | default | | 4352x4096x8192 | v9 | 1 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | 38 | | 3584x5376x8192 | v9 | 0 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | 38 | | 3584x5376x8192 | v9 | 1 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | 76 | | 3584x5376x8192 | v9 | 0 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | 76 | | 3584x5376x8192 | v9 | 1 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | default | | 3584x5376x8192 | v9 | 0 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | default | | 3584x5376x8192 | v9 | 1 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | 64 | | 4096x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | 64 | | 4096x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | 128 | | 4096x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | 128 | | 4096x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | default | | 4096x5120x8192 | v9 | 0 | 10 | 0 | 0 |
| v19_streamk_two_tile_reversed | default | | 4096x5120x8192 | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 4 | | 3328x5120x8192 | one_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 4 | | 3328x5120x8192 | one_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 4 | | 3328x5120x8192 | one_tile | owner | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 4 | | 3328x5120x8192 | one_tile | owner | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 4 | | 3328x5120x8192 | spread | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 4 | | 3328x5120x8192 | spread | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 4 | | 3328x5120x8192 | two_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 4 | | 3328x5120x8192 | two_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 8 | | 3328x5120x8192 | one_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 8 | | 3328x5120x8192 | one_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 8 | | 3328x5120x8192 | one_tile | owner | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 8 | | 3328x5120x8192 | one_tile | owner | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 8 | | 3328x5120x8192 | spread | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 8 | | 3328x5120x8192 | spread | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 8 | | 3328x5120x8192 | two_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 8 | | 3328x5120x8192 | two_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3328x5120x8192 | one_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3328x5120x8192 | one_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3328x5120x8192 | one_tile | owner | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3328x5120x8192 | one_tile | owner | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3328x5120x8192 | spread | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3328x5120x8192 | spread | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3328x5120x8192 | two_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3328x5120x8192 | two_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 16 | | 4352x4096x8192 | one_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 16 | | 4352x4096x8192 | one_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 16 | | 4352x4096x8192 | one_tile | owner | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 16 | | 4352x4096x8192 | one_tile | owner | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 16 | | 4352x4096x8192 | spread | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 16 | | 4352x4096x8192 | spread | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 16 | | 4352x4096x8192 | two_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 16 | | 4352x4096x8192 | two_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 32 | | 4352x4096x8192 | one_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 32 | | 4352x4096x8192 | one_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 32 | | 4352x4096x8192 | one_tile | owner | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 32 | | 4352x4096x8192 | one_tile | owner | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 32 | | 4352x4096x8192 | spread | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 32 | | 4352x4096x8192 | spread | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 32 | | 4352x4096x8192 | two_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 32 | | 4352x4096x8192 | two_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x8192 | one_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x8192 | one_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x8192 | one_tile | owner | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x8192 | one_tile | owner | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x8192 | spread | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x8192 | spread | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x8192 | two_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x8192 | two_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 38 | | 3584x5376x8192 | one_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 38 | | 3584x5376x8192 | one_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 38 | | 3584x5376x8192 | one_tile | owner | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 38 | | 3584x5376x8192 | one_tile | owner | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 38 | | 3584x5376x8192 | spread | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 38 | | 3584x5376x8192 | spread | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 38 | | 3584x5376x8192 | two_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 38 | | 3584x5376x8192 | two_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 76 | | 3584x5376x8192 | one_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 76 | | 3584x5376x8192 | one_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 76 | | 3584x5376x8192 | one_tile | owner | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 76 | | 3584x5376x8192 | one_tile | owner | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 76 | | 3584x5376x8192 | spread | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 76 | | 3584x5376x8192 | spread | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 76 | | 3584x5376x8192 | two_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 76 | | 3584x5376x8192 | two_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3584x5376x8192 | one_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3584x5376x8192 | one_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3584x5376x8192 | one_tile | owner | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3584x5376x8192 | one_tile | owner | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3584x5376x8192 | spread | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3584x5376x8192 | spread | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3584x5376x8192 | two_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3584x5376x8192 | two_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 64 | | 4096x5120x8192 | one_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 64 | | 4096x5120x8192 | one_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 64 | | 4096x5120x8192 | one_tile | owner | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 64 | | 4096x5120x8192 | one_tile | owner | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 64 | | 4096x5120x8192 | spread | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 64 | | 4096x5120x8192 | spread | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 64 | | 4096x5120x8192 | two_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 64 | | 4096x5120x8192 | two_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 128 | | 4096x5120x8192 | one_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 128 | | 4096x5120x8192 | one_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 128 | | 4096x5120x8192 | one_tile | owner | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 128 | | 4096x5120x8192 | one_tile | owner | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 128 | | 4096x5120x8192 | spread | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 128 | | 4096x5120x8192 | spread | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 128 | | 4096x5120x8192 | two_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 128 | | 4096x5120x8192 | two_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4096x5120x8192 | one_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4096x5120x8192 | one_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4096x5120x8192 | one_tile | owner | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4096x5120x8192 | one_tile | owner | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4096x5120x8192 | spread | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4096x5120x8192 | spread | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4096x5120x8192 | two_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4096x5120x8192 | two_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 76 | STREAMK_NUM_PROGRAMS=76 STREAMK_BALANCE_XCDS=0 | | 3584x5376x8192 | one_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 76 | STREAMK_NUM_PROGRAMS=76 STREAMK_BALANCE_XCDS=0 | | 3584x5376x8192 | one_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 76 | STREAMK_NUM_PROGRAMS=76 STREAMK_BALANCE_XCDS=0 | | 3584x5376x8192 | one_tile | owner | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 76 | STREAMK_NUM_PROGRAMS=76 STREAMK_BALANCE_XCDS=0 | | 3584x5376x8192 | one_tile | owner | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 76 | STREAMK_NUM_PROGRAMS=76 STREAMK_BALANCE_XCDS=0 | | 3584x5376x8192 | spread | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 76 | STREAMK_NUM_PROGRAMS=76 STREAMK_BALANCE_XCDS=0 | | 3584x5376x8192 | spread | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x512 | one_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x512 | one_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x512 | one_tile | owner | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x512 | one_tile | owner | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x512 | spread | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x512 | spread | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x512 | two_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x512 | two_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x1024 | one_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x1024 | one_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x1024 | one_tile | owner | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x1024 | one_tile | owner | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x1024 | spread | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x1024 | spread | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x1024 | two_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 4352x4096x1024 | two_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3584x5376x1024 | one_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3584x5376x1024 | one_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3584x5376x1024 | one_tile | owner | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3584x5376x1024 | one_tile | owner | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3584x5376x1024 | spread | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3584x5376x1024 | spread | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3584x5376x1024 | two_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | default | | 3584x5376x1024 | two_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 32 | STREAMK_NUM_PROGRAMS=32 | | 4352x4096x512 | one_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 32 | STREAMK_NUM_PROGRAMS=32 | | 4352x4096x512 | one_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 32 | STREAMK_NUM_PROGRAMS=32 | | 4352x4096x512 | one_tile | owner | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 32 | STREAMK_NUM_PROGRAMS=32 | | 4352x4096x512 | one_tile | owner | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 32 | STREAMK_NUM_PROGRAMS=32 | | 4352x4096x512 | spread | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 32 | STREAMK_NUM_PROGRAMS=32 | | 4352x4096x512 | spread | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 32 | STREAMK_NUM_PROGRAMS=32 | | 4352x4096x1024 | one_tile | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 32 | STREAMK_NUM_PROGRAMS=32 | | 4352x4096x1024 | one_tile | rs | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 32 | STREAMK_NUM_PROGRAMS=32 | | 4352x4096x1024 | one_tile | owner | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 32 | STREAMK_NUM_PROGRAMS=32 | | 4352x4096x1024 | one_tile | owner | v9 | 1 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 32 | STREAMK_NUM_PROGRAMS=32 | | 4352x4096x1024 | spread | rs | v9 | 0 | 10 | 0 | 0 |
| v20_streamk_reduce_scatter | 32 | STREAMK_NUM_PROGRAMS=32 | | 4352x4096x1024 | spread | rs | v9 | 1 | 10 | 0 | 0 |

PASS
