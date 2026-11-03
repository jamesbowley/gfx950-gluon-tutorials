# v31 entry-19 port: AMDGCN before and after

check_streamk's default and `--extra` cases, each launched with the kernel before the
port (v31's matmul_kernel.py with the edits reversed) and after, on the same inputs.
identical: same instruction stream (comments and debug lines stripped). bitwise:
outputs equal bit for bit.

240 cases: 240 bitwise equal, 240 match torch, 186 instruction-identical.

| tile | case | bitwise | identical | VGPRs | spills | instructions | v_cndmask | stores before | stores after |
|---|---|---|---|---|---|---|---|---|---|
| 64x64x64 | dp | True | False | 68 -> 60 | 0 -> 0 | 924 -> 897 | 1 -> 0 | dwordx4 4 | dwordx2 4 |
| 64x64x64 | capped | True | False | 80 -> 80 | 0 -> 0 | 1529 -> 1512 | 1 -> 0 | dwordx2 1, dwordx4 8 | dwordx2 5, dwordx4 4 |
| 64x64x64 | owner | True | False | 92 -> 96 | 0 -> 0 | 1687 -> 1653 | 2 -> 0 | dwordx4 12 | dwordx2 8, dwordx4 4 |
| 64x64x64 | rs | True | False | 84 -> 80 | 0 -> 0 | 1577 -> 1543 | 1 -> 0 | dwordx2 1, dwordx4 8 | dwordx2 5, dwordx4 4 |
| 64x64x64 | l2 | True | False | 84 -> 80 | 0 -> 0 | 1530 -> 1481 | 3 -> 2 | dwordx2 1, dwordx4 8, global_store_dword 2 | dwordx2 5, dwordx4 4, global_store_dword 2 |
| 64x64x64 | two_tile | True | False | 84 -> 80 | 0 -> 0 | 1177 -> 1161 | 1 -> 0 | dwordx4 8 | dwordx2 4, dwordx4 4 |
| 64x64x64 | spread | True | False | 92 -> 88 | 0 -> 0 | 3044 -> 2993 | 3 -> 0 | dwordx4 20 | dwordx2 12, dwordx4 8 |
| 64x64x64 | compact | True | False | 72 -> 68 | 0 -> 0 | 731 -> 716 | 1 -> 0 | dwordx4 8 | dwordx2 4, dwordx4 4 |
| 64x64x64 | l2_compact | True | False | 64 -> 64 | 0 -> 0 | 644 -> 644 | 2 -> 2 | dwordx2 1, dwordx4 4, global_store_dword 2 | same |
| 64x64x64 | l2_compact4 | True | False | 64 -> 64 | 0 -> 0 | 578 -> 578 | 2 -> 2 | dwordx2 1, dwordx4 4, global_store_dword 2 | same |
| 64x64x64 | compact_rs | True | True | 68 -> 68 | 0 -> 0 | 944 -> 944 | 0 -> 0 | dwordx2 2, dwordx4 4 | same |
| 64x64x64 | ragged | True | False | 96 -> 96 | 0 -> 0 | 1910 -> 1889 | 36 -> 36 | dwordx4 8, short 8 | dwordx2 4, dwordx4 4, short 8 |
| 64x64x64 | ragged_m | True | False | 280 -> 88 | 0 -> 0 | 1562 -> 1545 | 3 -> 3 | dwordx2 1, dwordx4 8 | dwordx2 5, dwordx4 4 |
| 64x64x64 | ragged_n | True | False | 96 -> 176 | 0 -> 0 | 1677 -> 1635 | 6 -> 6 | dwordx4 12 | dwordx2 8, dwordx4 4 |
| 64x64x64 | dp_below_wave | True | False | 68 -> 60 | 0 -> 0 | 924 -> 897 | 1 -> 0 | dwordx4 4 | dwordx2 4 |
| 64x64x64 | dp_single | True | False | 76 -> 68 | 0 -> 0 | 856 -> 874 | 1 -> 0 | dwordx4 4 | dwordx2 4 |
| 64x64x64 | dp_ragged_wave | True | False | 84 -> 84 | 0 -> 0 | 986 -> 986 | 18 -> 18 | dwordx4 4 | dwordx2 4 |
| 64x64x64 | kres_gt_dp | True | False | 72 -> 72 | 0 -> 0 | 939 -> 924 | 3 -> 2 | dwordx4 4 | dwordx2 4 |
| 64x64x64 | kres_gt_rs | True | False | 168 -> 84 | 0 -> 0 | 1586 -> 1487 | 7 -> 6 | dwordx2 1, dwordx4 8 | dwordx2 5, dwordx4 4 |
| 64x64x64 | kres_eq_dp | True | False | 64 -> 68 | 0 -> 0 | 896 -> 878 | 1 -> 0 | dwordx4 4 | dwordx2 4 |
| 64x64x64 | kres_eq_owner | True | False | 100 -> 104 | 0 -> 0 | 1773 -> 1697 | 6 -> 4 | dwordx4 12 | dwordx2 8, dwordx4 4 |
| 64x64x64 | dp_1wave | True | False | 68 -> 60 | 0 -> 0 | 924 -> 897 | 1 -> 0 | dwordx4 4 | dwordx2 4 |
| 64x64x64 | dp_4wave | True | False | 76 -> 68 | 0 -> 0 | 955 -> 948 | 1 -> 0 | dwordx4 4 | dwordx2 4 |
| 64x64x64 | dp_4w_kres | True | False | 64 -> 72 | 0 -> 0 | 900 -> 882 | 3 -> 2 | dwordx4 4 | dwordx2 4 |
| 64x64x64 | rs_3wave | True | False | 84 -> 80 | 0 -> 0 | 1547 -> 1499 | 1 -> 0 | dwordx2 1, dwordx4 8 | dwordx2 5, dwordx4 4 |
| 64x64x64 | rs_3w_kres | True | False | 88 -> 88 | 0 -> 0 | 1571 -> 1541 | 7 -> 6 | dwordx2 1, dwordx4 8 | dwordx2 5, dwordx4 4 |
| 64x64x64 | l2_3w_kres | True | False | 168 -> 88 | 0 -> 0 | 1568 -> 1469 | 7 -> 6 | dwordx2 1, dwordx4 8, global_store_dword 2 | dwordx2 5, dwordx4 4, global_store_dword 2 |
| 64x64x64 | two_tile_2w | True | False | 92 -> 88 | 0 -> 0 | 3050 -> 2999 | 3 -> 0 | dwordx4 20 | dwordx2 12, dwordx4 8 |
| 64x64x64 | two_tile_kres | True | False | 252 -> 96 | 0 -> 0 | 3150 -> 3112 | 21 -> 21 | dwordx4 20 | dwordx2 12, dwordx4 8 |
| 64x64x64 | spread_2w | True | False | 92 -> 88 | 0 -> 0 | 3044 -> 2993 | 3 -> 0 | dwordx4 20 | dwordx2 12, dwordx4 8 |
| 128x64x64 | dp | True | True | 112 -> 112 | 0 -> 0 | 1038 -> 1038 | 0 -> 0 | dwordx4 4 | same |
| 128x64x64 | capped | True | True | 120 -> 120 | 0 -> 0 | 1801 -> 1801 | 0 -> 0 | dwordx2 1, dwordx4 12 | same |
| 128x64x64 | owner | True | True | 164 -> 164 | 0 -> 0 | 1996 -> 1996 | 0 -> 0 | dwordx4 16 | same |
| 128x64x64 | rs | True | True | 128 -> 128 | 0 -> 0 | 2042 -> 2042 | 0 -> 0 | dwordx2 2, dwordx4 12 | same |
| 128x64x64 | l2 | True | False | 116 -> 116 | 0 -> 0 | 1739 -> 1733 | 2 -> 2 | dwordx2 2, dwordx4 12, global_store_dword 2 | dwordx4 13, global_store_dword 2 |
| 128x64x64 | two_tile | True | True | 144 -> 144 | 0 -> 0 | 1345 -> 1345 | 0 -> 0 | dwordx4 12 | same |
| 128x64x64 | spread | True | True | 152 -> 152 | 0 -> 0 | 3459 -> 3459 | 0 -> 0 | dwordx4 28 | same |
| 128x64x64 | compact | True | True | 112 -> 112 | 0 -> 0 | 868 -> 868 | 0 -> 0 | dwordx4 12 | same |
| 128x64x64 | l2_compact | True | False | 96 -> 96 | 0 -> 0 | 764 -> 759 | 2 -> 2 | dwordx2 2, dwordx4 8, global_store_dword 2 | dwordx4 9, global_store_dword 2 |
| 128x64x64 | l2_compact4 | True | False | 96 -> 96 | 0 -> 0 | 729 -> 724 | 2 -> 2 | dwordx2 2, dwordx4 8, global_store_dword 2 | dwordx4 9, global_store_dword 2 |
| 128x64x64 | compact_rs | True | False | 100 -> 100 | 0 -> 0 | 1055 -> 1049 | 0 -> 0 | dwordx2 4, dwordx4 8 | dwordx4 10 |
| 128x64x64 | ragged | True | True | 148 -> 148 | 0 -> 0 | 2148 -> 2148 | 42 -> 42 | dwordx4 12, short 8 | same |
| 128x64x64 | ragged_m | True | True | 128 -> 128 | 0 -> 0 | 2057 -> 2057 | 4 -> 4 | dwordx2 2, dwordx4 12 | same |
| 128x64x64 | ragged_n | True | True | 160 -> 160 | 0 -> 0 | 2009 -> 2009 | 6 -> 6 | dwordx4 16 | same |
| 128x64x64 | dp_below_wave | True | True | 112 -> 112 | 0 -> 0 | 942 -> 942 | 0 -> 0 | dwordx4 4 | same |
| 128x64x64 | dp_single | True | True | 116 -> 116 | 0 -> 0 | 983 -> 983 | 0 -> 0 | dwordx4 4 | same |
| 128x64x64 | dp_ragged_wave | True | True | 124 -> 124 | 0 -> 0 | 1021 -> 1021 | 22 -> 22 | dwordx4 4 | same |
| 128x64x64 | kres_gt_dp | True | True | 112 -> 112 | 0 -> 0 | 1019 -> 1019 | 3 -> 3 | dwordx4 4 | same |
| 128x64x64 | kres_gt_rs | True | True | 120 -> 120 | 0 -> 0 | 1921 -> 1921 | 9 -> 9 | dwordx2 2, dwordx4 12 | same |
| 128x64x64 | kres_eq_dp | True | True | 104 -> 104 | 0 -> 0 | 966 -> 966 | 0 -> 0 | dwordx4 4 | same |
| 128x64x64 | kres_eq_owner | True | True | 160 -> 160 | 0 -> 0 | 2018 -> 2018 | 6 -> 6 | dwordx4 16 | same |
| 128x64x64 | dp_1wave | True | True | 112 -> 112 | 0 -> 0 | 1038 -> 1038 | 0 -> 0 | dwordx4 4 | same |
| 128x64x64 | dp_4wave | True | True | 116 -> 116 | 0 -> 0 | 1080 -> 1080 | 0 -> 0 | dwordx4 4 | same |
| 128x64x64 | dp_4w_kres | True | True | 108 -> 108 | 0 -> 0 | 971 -> 971 | 3 -> 3 | dwordx4 4 | same |
| 128x64x64 | rs_3wave | True | True | 120 -> 120 | 0 -> 0 | 1966 -> 1966 | 0 -> 0 | dwordx2 2, dwordx4 12 | same |
| 128x64x64 | rs_3w_kres | True | True | 120 -> 120 | 0 -> 0 | 2007 -> 2007 | 9 -> 9 | dwordx2 2, dwordx4 12 | same |
| 128x64x64 | l2_3w_kres | True | False | 116 -> 116 | 0 -> 0 | 1689 -> 1686 | 8 -> 8 | dwordx2 2, dwordx4 12, global_store_dword 2 | dwordx4 13, global_store_dword 2 |
| 128x64x64 | two_tile_2w | True | True | 152 -> 152 | 0 -> 0 | 3465 -> 3465 | 0 -> 0 | dwordx4 28 | same |
| 128x64x64 | two_tile_kres | True | True | 152 -> 152 | 0 -> 0 | 3571 -> 3571 | 31 -> 31 | dwordx4 28 | same |
| 128x64x64 | spread_2w | True | True | 152 -> 152 | 0 -> 0 | 3459 -> 3459 | 0 -> 0 | dwordx4 28 | same |
| 64x128x64 | dp | True | True | 116 -> 116 | 0 -> 0 | 1048 -> 1048 | 0 -> 0 | dwordx4 4 | same |
| 64x128x64 | capped | True | True | 124 -> 124 | 0 -> 0 | 1803 -> 1803 | 0 -> 0 | dwordx2 1, dwordx4 12 | same |
| 64x128x64 | owner | True | True | 164 -> 164 | 0 -> 0 | 2016 -> 2016 | 0 -> 0 | dwordx4 16 | same |
| 64x128x64 | rs | True | True | 124 -> 124 | 0 -> 0 | 2080 -> 2080 | 0 -> 0 | dwordx2 2, dwordx4 12 | same |
| 64x128x64 | l2 | True | True | 120 -> 120 | 0 -> 0 | 1747 -> 1747 | 2 -> 2 | dwordx4 13, global_store_dword 2 | same |
| 64x128x64 | two_tile | True | True | 144 -> 144 | 0 -> 0 | 1356 -> 1356 | 0 -> 0 | dwordx4 12 | same |
| 64x128x64 | spread | True | True | 152 -> 152 | 0 -> 0 | 3484 -> 3484 | 0 -> 0 | dwordx4 28 | same |
| 64x128x64 | compact | True | True | 112 -> 112 | 0 -> 0 | 913 -> 913 | 0 -> 0 | dwordx4 12 | same |
| 64x128x64 | l2_compact | True | True | 100 -> 100 | 0 -> 0 | 812 -> 812 | 2 -> 2 | dwordx4 9, global_store_dword 2 | same |
| 64x128x64 | l2_compact4 | True | True | 100 -> 100 | 0 -> 0 | 730 -> 730 | 2 -> 2 | dwordx4 9, global_store_dword 2 | same |
| 64x128x64 | compact_rs | True | True | 100 -> 100 | 0 -> 0 | 1147 -> 1147 | 0 -> 0 | dwordx4 10 | same |
| 64x128x64 | ragged | True | True | 152 -> 152 | 0 -> 0 | 2162 -> 2162 | 42 -> 42 | dwordx2 2, dwordx4 12 | same |
| 64x128x64 | ragged_m | True | True | 132 -> 132 | 0 -> 0 | 2078 -> 2078 | 4 -> 4 | dwordx2 2, dwordx4 12 | same |
| 64x128x64 | ragged_n | True | True | 156 -> 156 | 0 -> 0 | 2047 -> 2047 | 6 -> 6 | dwordx4 16 | same |
| 64x128x64 | dp_below_wave | True | True | 116 -> 116 | 0 -> 0 | 1048 -> 1048 | 0 -> 0 | dwordx4 4 | same |
| 64x128x64 | dp_single | True | True | 112 -> 112 | 0 -> 0 | 1026 -> 1026 | 0 -> 0 | dwordx4 4 | same |
| 64x128x64 | dp_ragged_wave | True | True | 132 -> 132 | 0 -> 0 | 1171 -> 1171 | 28 -> 28 | dwordx4 4 | same |
| 64x128x64 | kres_gt_dp | True | True | 116 -> 116 | 0 -> 0 | 1061 -> 1061 | 3 -> 3 | dwordx4 4 | same |
| 64x128x64 | kres_gt_rs | True | True | 120 -> 120 | 0 -> 0 | 1921 -> 1921 | 9 -> 9 | dwordx2 2, dwordx4 12 | same |
| 64x128x64 | kres_eq_dp | True | True | 104 -> 104 | 0 -> 0 | 975 -> 975 | 0 -> 0 | dwordx4 4 | same |
| 64x128x64 | kres_eq_owner | True | True | 164 -> 164 | 0 -> 0 | 2102 -> 2102 | 6 -> 6 | dwordx4 16 | same |
| 64x128x64 | dp_1wave | True | True | 116 -> 116 | 0 -> 0 | 1048 -> 1048 | 0 -> 0 | dwordx4 4 | same |
| 64x128x64 | dp_4wave | True | True | 112 -> 112 | 0 -> 0 | 1124 -> 1124 | 0 -> 0 | dwordx4 4 | same |
| 64x128x64 | dp_4w_kres | True | True | 108 -> 108 | 0 -> 0 | 980 -> 980 | 3 -> 3 | dwordx4 4 | same |
| 64x128x64 | rs_3wave | True | True | 124 -> 124 | 0 -> 0 | 1967 -> 1967 | 0 -> 0 | dwordx2 2, dwordx4 12 | same |
| 64x128x64 | rs_3w_kres | True | True | 128 -> 128 | 0 -> 0 | 2042 -> 2042 | 9 -> 9 | dwordx2 2, dwordx4 12 | same |
| 64x128x64 | l2_3w_kres | True | True | 116 -> 116 | 0 -> 0 | 1696 -> 1696 | 8 -> 8 | dwordx4 13, global_store_dword 2 | same |
| 64x128x64 | two_tile_2w | True | True | 152 -> 152 | 0 -> 0 | 3490 -> 3490 | 0 -> 0 | dwordx4 28 | same |
| 64x128x64 | two_tile_kres | True | True | 152 -> 152 | 0 -> 0 | 3674 -> 3674 | 26 -> 26 | dwordx4 28 | same |
| 64x128x64 | spread_2w | True | True | 152 -> 152 | 0 -> 0 | 3484 -> 3484 | 0 -> 0 | dwordx4 28 | same |
| 128x128x64 | dp | True | True | 184 -> 184 | 0 -> 0 | 1374 -> 1374 | 0 -> 0 | dwordx4 8 | same |
| 128x128x64 | capped | True | True | 196 -> 196 | 0 -> 0 | 2559 -> 2559 | 0 -> 0 | dwordx2 2, dwordx4 24 | same |
| 128x128x64 | owner | True | True | 320 -> 320 | 0 -> 0 | 2867 -> 2867 | 0 -> 0 | dwordx4 32 | same |
| 128x128x64 | rs | True | True | 204 -> 204 | 0 -> 0 | 2524 -> 2524 | 0 -> 0 | dwordx2 2, dwordx4 24 | same |
| 128x128x64 | l2 | True | True | 192 -> 192 | 0 -> 0 | 2257 -> 2257 | 2 -> 2 | dwordx4 26, global_store_dword 2 | same |
| 128x128x64 | two_tile | True | True | 304 -> 304 | 0 -> 0 | 1849 -> 1849 | 0 -> 0 | dwordx4 24 | same |
| 128x128x64 | spread | True | True | 332 -> 332 | 0 -> 0 | 4779 -> 4779 | 0 -> 0 | dwordx4 56 | same |
| 128x128x64 | compact | True | True | 172 -> 172 | 0 -> 0 | 1177 -> 1177 | 0 -> 0 | dwordx4 24 | same |
| 128x128x64 | l2_compact | True | True | 148 -> 148 | 0 -> 0 | 1026 -> 1026 | 2 -> 2 | dwordx4 18, global_store_dword 2 | same |
| 128x128x64 | l2_compact4 | True | True | 148 -> 148 | 0 -> 0 | 919 -> 919 | 2 -> 2 | dwordx4 18, global_store_dword 2 | same |
| 128x128x64 | compact_rs | True | True | 152 -> 152 | 0 -> 0 | 1382 -> 1382 | 0 -> 0 | dwordx4 20 | same |
| 128x128x64 | ragged | True | True | 204 -> 204 | 0 -> 0 | 2682 -> 2682 | 52 -> 52 | dwordx2 2, dwordx4 24 | same |
| 128x128x64 | ragged_m | True | True | 204 -> 204 | 0 -> 0 | 2549 -> 2549 | 6 -> 6 | dwordx2 2, dwordx4 24 | same |
| 128x128x64 | ragged_n | True | True | 320 -> 320 | 0 -> 0 | 2995 -> 2995 | 12 -> 12 | dwordx4 32 | same |
| 128x128x64 | dp_below_wave | True | True | 184 -> 184 | 0 -> 0 | 1374 -> 1374 | 0 -> 0 | dwordx4 8 | same |
| 128x128x64 | dp_single | True | True | 192 -> 192 | 0 -> 0 | 1334 -> 1334 | 0 -> 0 | dwordx4 8 | same |
| 128x128x64 | dp_ragged_wave | True | True | 216 -> 216 | 0 -> 0 | 1515 -> 1515 | 36 -> 36 | dwordx4 8 | same |
| 128x128x64 | kres_gt_dp | True | True | 192 -> 192 | 0 -> 0 | 1361 -> 1361 | 4 -> 4 | dwordx4 8 | same |
| 128x128x64 | kres_gt_rs | True | True | 184 -> 184 | 0 -> 0 | 2342 -> 2342 | 12 -> 12 | dwordx2 2, dwordx4 24 | same |
| 128x128x64 | kres_eq_dp | True | True | 168 -> 168 | 0 -> 0 | 1262 -> 1262 | 0 -> 0 | dwordx4 8 | same |
| 128x128x64 | kres_eq_owner | True | True | 316 -> 316 | 0 -> 0 | 2929 -> 2929 | 8 -> 8 | dwordx4 32 | same |
| 128x128x64 | dp_1wave | True | True | 184 -> 184 | 0 -> 0 | 1374 -> 1374 | 0 -> 0 | dwordx4 8 | same |
| 128x128x64 | dp_4wave | True | True | 192 -> 192 | 0 -> 0 | 1431 -> 1431 | 0 -> 0 | dwordx4 8 | same |
| 128x128x64 | dp_4w_kres | True | True | 172 -> 172 | 0 -> 0 | 1268 -> 1268 | 4 -> 4 | dwordx4 8 | same |
| 128x128x64 | rs_3wave | True | True | 196 -> 196 | 0 -> 0 | 2435 -> 2435 | 0 -> 0 | dwordx2 2, dwordx4 24 | same |
| 128x128x64 | rs_3w_kres | True | True | 200 -> 200 | 0 -> 0 | 2476 -> 2476 | 12 -> 12 | dwordx2 2, dwordx4 24 | same |
| 128x128x64 | l2_3w_kres | True | True | 180 -> 180 | 0 -> 0 | 2162 -> 2162 | 10 -> 10 | dwordx4 26, global_store_dword 2 | same |
| 128x128x64 | two_tile_2w | True | True | 332 -> 332 | 0 -> 0 | 4782 -> 4782 | 0 -> 0 | dwordx4 56 | same |
| 128x128x64 | two_tile_kres | True | True | 320 -> 320 | 0 -> 0 | 4996 -> 4996 | 42 -> 42 | dwordx4 56 | same |
| 128x128x64 | spread_2w | True | True | 332 -> 332 | 0 -> 0 | 4779 -> 4779 | 0 -> 0 | dwordx4 56 | same |
| 128x128x128 | dp | True | True | 252 -> 252 | 0 -> 0 | 1836 -> 1836 | 0 -> 0 | dwordx4 8 | same |
| 128x128x128 | capped | True | True | 260 -> 260 | 0 -> 0 | 3274 -> 3274 | 0 -> 0 | dwordx2 2, dwordx4 24 | same |
| 128x128x128 | owner | True | True | 340 -> 340 | 0 -> 0 | 3484 -> 3484 | 0 -> 0 | dwordx4 32 | same |
| 128x128x128 | rs | True | True | 256 -> 256 | 0 -> 0 | 3257 -> 3257 | 0 -> 0 | dwordx2 2, dwordx4 24 | same |
| 128x128x128 | l2 | True | True | 252 -> 252 | 0 -> 0 | 2960 -> 2960 | 2 -> 2 | dwordx4 26, global_store_dword 2 | same |
| 128x128x128 | two_tile | True | True | 284 -> 284 | 0 -> 0 | 2136 -> 2136 | 0 -> 0 | dwordx4 24 | same |
| 128x128x128 | spread | True | True | 292 -> 292 | 0 -> 0 | 5785 -> 5785 | 0 -> 0 | dwordx4 56 | same |
| 128x128x128 | compact | True | True | 220 -> 220 | 0 -> 0 | 1474 -> 1474 | 0 -> 0 | dwordx4 24 | same |
| 128x128x128 | l2_compact | True | True | 224 -> 224 | 0 -> 0 | 1348 -> 1348 | 2 -> 2 | dwordx4 18, global_store_dword 2 | same |
| 128x128x128 | l2_compact4 | True | True | 224 -> 224 | 0 -> 0 | 1241 -> 1241 | 2 -> 2 | dwordx4 18, global_store_dword 2 | same |
| 128x128x128 | compact_rs | True | True | 220 -> 220 | 0 -> 0 | 1713 -> 1713 | 0 -> 0 | dwordx4 20 | same |
| 128x128x128 | ragged | True | True | 284 -> 284 | 0 -> 0 | 3543 -> 3543 | 76 -> 76 | dwordx2 2, dwordx4 24 | same |
| 128x128x128 | ragged_m | True | True | 284 -> 284 | 0 -> 0 | 3353 -> 3353 | 6 -> 6 | dwordx2 2, dwordx4 24 | same |
| 128x128x128 | ragged_n | True | False | 348 -> 348 | 0 -> 0 | 3625 -> 3625 | 12 -> 12 | dwordx4 32 | same |
| 128x128x128 | dp_below_wave | True | True | 252 -> 252 | 0 -> 0 | 1836 -> 1836 | 0 -> 0 | dwordx4 8 | same |
| 128x128x128 | dp_single | True | True | 260 -> 260 | 0 -> 0 | 1811 -> 1811 | 0 -> 0 | dwordx4 8 | same |
| 128x128x128 | dp_ragged_wave | True | True | 284 -> 284 | 0 -> 0 | 2083 -> 2083 | 52 -> 52 | dwordx4 8 | same |
| 128x128x128 | kres_gt_dp | True | True | 264 -> 264 | 0 -> 0 | 1834 -> 1834 | 8 -> 8 | dwordx4 8 | same |
| 128x128x128 | kres_gt_rs | True | True | 260 -> 260 | 0 -> 0 | 3026 -> 3026 | 24 -> 24 | dwordx2 2, dwordx4 24 | same |
| 128x128x128 | kres_eq_dp | True | True | 240 -> 240 | 0 -> 0 | 1757 -> 1757 | 0 -> 0 | dwordx4 8 | same |
| 128x128x128 | kres_eq_owner | True | False | 336 -> 336 | 0 -> 0 | 3488 -> 3488 | 16 -> 16 | dwordx4 32 | same |
| 128x128x128 | dp_1wave | True | True | 252 -> 252 | 0 -> 0 | 1836 -> 1836 | 0 -> 0 | dwordx4 8 | same |
| 128x128x128 | dp_4wave | True | True | 260 -> 260 | 0 -> 0 | 1908 -> 1908 | 0 -> 0 | dwordx4 8 | same |
| 128x128x128 | dp_4w_kres | True | True | 248 -> 248 | 0 -> 0 | 1767 -> 1767 | 8 -> 8 | dwordx4 8 | same |
| 128x128x128 | rs_3wave | True | True | 252 -> 252 | 0 -> 0 | 3142 -> 3142 | 0 -> 0 | dwordx2 2, dwordx4 24 | same |
| 128x128x128 | rs_3w_kres | True | True | 260 -> 260 | 0 -> 0 | 3132 -> 3132 | 24 -> 24 | dwordx2 2, dwordx4 24 | same |
| 128x128x128 | l2_3w_kres | True | True | 256 -> 256 | 0 -> 0 | 2837 -> 2837 | 18 -> 18 | dwordx4 26, global_store_dword 2 | same |
| 128x128x128 | two_tile_2w | True | True | 292 -> 292 | 0 -> 0 | 5718 -> 5718 | 0 -> 0 | dwordx4 56 | same |
| 128x128x128 | two_tile_kres | True | False | 300 -> 300 | 0 -> 0 | 6083 -> 6083 | 72 -> 72 | dwordx4 56 | same |
| 128x128x128 | spread_2w | True | True | 292 -> 292 | 0 -> 0 | 5785 -> 5785 | 0 -> 0 | dwordx4 56 | same |
| 256x128x64 | dp | True | True | 328 -> 328 | 0 -> 0 | 1984 -> 1984 | 0 -> 0 | dwordx4 16 | same |
| 256x128x64 | capped | True | True | 292 -> 292 | 0 -> 0 | 3506 -> 3506 | 0 -> 0 | dwordx4 50 | same |
| 256x128x64 | owner | True | True | 440 -> 440 | 0 -> 0 | 4235 -> 4235 | 0 -> 0 | dwordx4 64 | same |
| 256x128x64 | rs | True | True | 296 -> 296 | 0 -> 0 | 3497 -> 3497 | 0 -> 0 | dwordx4 50 | same |
| 256x128x64 | l2 | True | True | 288 -> 288 | 0 -> 0 | 3225 -> 3225 | 42 -> 42 | dwordx4 52, global_store_dword 2 | same |
| 256x128x64 | two_tile | True | True | 384 -> 384 | 0 -> 0 | 2818 -> 2818 | 0 -> 0 | dwordx4 48 | same |
| 256x128x64 | spread | True | True | 404 -> 404 | 0 -> 0 | 7397 -> 7397 | 0 -> 0 | dwordx4 112 | same |
| 256x128x64 | compact | True | True | 252 -> 252 | 0 -> 0 | 1655 -> 1655 | 0 -> 0 | dwordx4 48 | same |
| 256x128x64 | l2_compact | True | True | 252 -> 252 | 0 -> 0 | 1450 -> 1450 | 42 -> 42 | dwordx4 36, global_store_dword 2 | same |
| 256x128x64 | l2_compact4 | True | True | 256 -> 256 | 0 -> 0 | 1321 -> 1321 | 42 -> 42 | dwordx4 36, global_store_dword 2 | same |
| 256x128x64 | compact_rs | True | True | 252 -> 252 | 0 -> 0 | 1798 -> 1798 | 0 -> 0 | dwordx4 40 | same |
| 256x128x64 | ragged | True | True | 332 -> 332 | 0 -> 0 | 3778 -> 3778 | 78 -> 78 | dwordx4 50 | same |
| 256x128x64 | ragged_m | True | True | 428 -> 428 | 0 -> 0 | 3653 -> 3653 | 10 -> 10 | dwordx4 50 | same |
| 256x128x64 | ragged_n | True | True | 432 -> 432 | 0 -> 0 | 4371 -> 4371 | 24 -> 24 | dwordx4 64 | same |
| 256x128x64 | dp_below_wave | True | True | 328 -> 328 | 0 -> 0 | 1888 -> 1888 | 0 -> 0 | dwordx4 16 | same |
| 256x128x64 | dp_single | True | True | 336 -> 336 | 0 -> 0 | 1972 -> 1972 | 0 -> 0 | dwordx4 16 | same |
| 256x128x64 | dp_ragged_wave | True | True | 324 -> 324 | 0 -> 0 | 2090 -> 2090 | 52 -> 52 | dwordx4 16 | same |
| 256x128x64 | kres_gt_dp | True | True | 316 -> 316 | 0 -> 0 | 1945 -> 1945 | 6 -> 6 | dwordx4 16 | same |
| 256x128x64 | kres_gt_rs | True | True | 292 -> 292 | 0 -> 0 | 3197 -> 3197 | 18 -> 18 | dwordx4 50 | same |
| 256x128x64 | kres_eq_dp | True | True | 300 -> 300 | 0 -> 0 | 1855 -> 1855 | 0 -> 0 | dwordx4 16 | same |
| 256x128x64 | kres_eq_owner | True | True | 432 -> 432 | 0 -> 0 | 4176 -> 4176 | 12 -> 12 | dwordx4 64 | same |
| 256x128x64 | dp_1wave | True | True | 328 -> 328 | 0 -> 0 | 1984 -> 1984 | 0 -> 0 | dwordx4 16 | same |
| 256x128x64 | dp_4wave | True | True | 336 -> 336 | 0 -> 0 | 2068 -> 2068 | 0 -> 0 | dwordx4 16 | same |
| 256x128x64 | dp_4w_kres | True | True | 304 -> 304 | 0 -> 0 | 1863 -> 1863 | 6 -> 6 | dwordx4 16 | same |
| 256x128x64 | rs_3wave | True | True | 288 -> 288 | 0 -> 0 | 3354 -> 3354 | 0 -> 0 | dwordx4 50 | same |
| 256x128x64 | rs_3w_kres | True | True | 292 -> 292 | 0 -> 0 | 3328 -> 3328 | 18 -> 18 | dwordx4 50 | same |
| 256x128x64 | l2_3w_kres | True | True | 292 -> 292 | 0 -> 0 | 3056 -> 3056 | 54 -> 54 | dwordx4 52, global_store_dword 2 | same |
| 256x128x64 | two_tile_2w | True | True | 404 -> 404 | 0 -> 0 | 7268 -> 7268 | 0 -> 0 | dwordx4 112 | same |
| 256x128x64 | two_tile_kres | True | True | 428 -> 428 | 0 -> 0 | 7748 -> 7748 | 90 -> 90 | dwordx4 112 | same |
| 256x128x64 | spread_2w | True | True | 404 -> 404 | 0 -> 0 | 7397 -> 7397 | 0 -> 0 | dwordx4 112 | same |
| 128x256x64 | dp | True | True | 332 -> 332 | 0 -> 0 | 1968 -> 1968 | 0 -> 0 | dwordx4 16 | same |
| 128x256x64 | capped | True | False | 372 -> 372 | 0 -> 0 | 3569 -> 3566 | 0 -> 0 | dwordx4 50 | same |
| 128x256x64 | owner | True | True | 440 -> 440 | 0 -> 0 | 4223 -> 4223 | 0 -> 0 | dwordx4 64 | same |
| 128x256x64 | rs | True | False | 316 -> 316 | 0 -> 0 | 3568 -> 3568 | 0 -> 0 | dwordx4 50 | same |
| 128x256x64 | l2 | True | True | 296 -> 296 | 0 -> 0 | 3208 -> 3208 | 42 -> 42 | dwordx4 52, global_store_dword 2 | same |
| 128x256x64 | two_tile | True | True | 384 -> 384 | 0 -> 0 | 2791 -> 2791 | 0 -> 0 | dwordx4 48 | same |
| 128x256x64 | spread | True | True | 404 -> 404 | 0 -> 0 | 7219 -> 7219 | 0 -> 0 | dwordx4 112 | same |
| 128x256x64 | compact | True | True | 256 -> 256 | 0 -> 0 | 1686 -> 1686 | 0 -> 0 | dwordx4 48 | same |
| 128x256x64 | l2_compact | True | False | 252 -> 252 | 0 -> 0 | 1521 -> 1531 | 42 -> 42 | dwordx4 36, global_store_dword 2 | same |
| 128x256x64 | l2_compact4 | True | True | 256 -> 256 | 0 -> 0 | 1323 -> 1323 | 42 -> 42 | dwordx4 36, global_store_dword 2 | same |
| 128x256x64 | compact_rs | True | False | 256 -> 256 | 0 -> 0 | 1920 -> 1914 | 0 -> 0 | dwordx4 40 | same |
| 128x256x64 | ragged | True | False | 348 -> 348 | 0 -> 0 | 3868 -> 3869 | 90 -> 90 | dwordx4 50 | same |
| 128x256x64 | ragged_m | True | False | 396 -> 396 | 0 -> 0 | 3694 -> 3694 | 10 -> 10 | dwordx4 50 | same |
| 128x256x64 | ragged_n | True | True | 436 -> 436 | 0 -> 0 | 4371 -> 4371 | 24 -> 24 | dwordx4 64 | same |
| 128x256x64 | dp_below_wave | True | True | 332 -> 332 | 0 -> 0 | 1968 -> 1968 | 0 -> 0 | dwordx4 16 | same |
| 128x256x64 | dp_single | True | True | 356 -> 356 | 0 -> 0 | 2019 -> 2019 | 0 -> 0 | dwordx4 16 | same |
| 128x256x64 | dp_ragged_wave | True | True | 352 -> 352 | 0 -> 0 | 2313 -> 2313 | 64 -> 64 | dwordx4 16 | same |
| 128x256x64 | kres_gt_dp | True | True | 340 -> 340 | 0 -> 0 | 2003 -> 2003 | 6 -> 6 | dwordx4 16 | same |
| 128x256x64 | kres_gt_rs | True | False | 300 -> 300 | 0 -> 0 | 3213 -> 3209 | 18 -> 18 | dwordx4 50 | same |
| 128x256x64 | kres_eq_dp | True | True | 296 -> 296 | 0 -> 0 | 1828 -> 1828 | 0 -> 0 | dwordx4 16 | same |
| 128x256x64 | kres_eq_owner | True | True | 436 -> 436 | 0 -> 0 | 4295 -> 4295 | 12 -> 12 | dwordx4 64 | same |
| 128x256x64 | dp_1wave | True | True | 332 -> 332 | 0 -> 0 | 1968 -> 1968 | 0 -> 0 | dwordx4 16 | same |
| 128x256x64 | dp_4wave | True | True | 356 -> 356 | 0 -> 0 | 2115 -> 2115 | 0 -> 0 | dwordx4 16 | same |
| 128x256x64 | dp_4w_kres | True | True | 304 -> 304 | 0 -> 0 | 1836 -> 1836 | 6 -> 6 | dwordx4 16 | same |
| 128x256x64 | rs_3wave | True | False | 296 -> 296 | 0 -> 0 | 3357 -> 3353 | 0 -> 0 | dwordx4 50 | same |
| 128x256x64 | rs_3w_kres | True | False | 312 -> 312 | 0 -> 0 | 3426 -> 3428 | 18 -> 18 | dwordx4 50 | same |
| 128x256x64 | l2_3w_kres | True | True | 300 -> 300 | 0 -> 0 | 3045 -> 3045 | 54 -> 54 | dwordx4 52, global_store_dword 2 | same |
| 128x256x64 | two_tile_2w | True | True | 404 -> 404 | 0 -> 0 | 7229 -> 7229 | 0 -> 0 | dwordx4 112 | same |
| 128x256x64 | two_tile_kres | True | True | 420 -> 420 | 0 -> 0 | 7901 -> 7901 | 80 -> 80 | dwordx4 112 | same |
| 128x256x64 | spread_2w | True | True | 404 -> 404 | 0 -> 0 | 7219 -> 7219 | 0 -> 0 | dwordx4 112 | same |
| 256x256x64 | dp | True | True | 464 -> 464 | 0 -> 0 | 3394 -> 3394 | 0 -> 0 | dwordx4 32 | same |
| 256x256x64 | capped | True | True | 484 -> 484 | 0 -> 0 | 5073 -> 5073 | 0 -> 0 | dwordx4 100 | same |
| 256x256x64 | owner | True | True | 512 -> 512 | 2 -> 2 | 6933 -> 6933 | 0 -> 0 | dwordx4 128 | same |
| 256x256x64 | rs | True | False | 492 -> 492 | 0 -> 0 | 5199 -> 5199 | 0 -> 0 | dwordx4 100 | same |
| 256x256x64 | l2 | True | True | 480 -> 480 | 0 -> 0 | 4800 -> 4800 | 2 -> 2 | dwordx4 104, global_store_dword 2 | same |
| 256x256x64 | two_tile | True | True | 512 -> 512 | 7 -> 7 | 4573 -> 4573 | 0 -> 0 | dwordx4 96 | same |
| 256x256x64 | spread | True | True | 512 -> 512 | 15 -> 15 | 12099 -> 12099 | 0 -> 0 | dwordx4 224 | same |
| 256x256x64 | compact | True | True | 416 -> 416 | 0 -> 0 | 2963 -> 2963 | 0 -> 0 | dwordx4 96 | same |
| 256x256x64 | l2_compact | True | False | 432 -> 432 | 0 -> 0 | 2305 -> 2322 | 2 -> 2 | dwordx4 72, global_store_dword 2 | same |
| 256x256x64 | l2_compact4 | True | True | 424 -> 424 | 0 -> 0 | 1965 -> 1965 | 2 -> 2 | dwordx4 72, global_store_dword 2 | same |
| 256x256x64 | compact_rs | True | False | 448 -> 444 | 0 -> 0 | 2825 -> 2818 | 0 -> 0 | dwordx4 80 | same |
| 256x256x64 | ragged | True | False | 512 -> 512 | 6 -> 6 | 5667 -> 5669 | 132 -> 132 | dwordx4 100 | same |
| 256x256x64 | ragged_m | True | False | 512 -> 512 | 17 -> 17 | 5658 -> 5658 | 20 -> 20 | dwordx4 100 | same |
| 256x256x64 | ragged_n | True | True | 512 -> 512 | 3 -> 3 | 7416 -> 7416 | 48 -> 48 | dwordx4 128 | same |
| 256x256x64 | dp_below_wave | True | True | 464 -> 464 | 0 -> 0 | 3394 -> 3394 | 0 -> 0 | dwordx4 32 | same |
| 256x256x64 | dp_single | True | True | 464 -> 464 | 0 -> 0 | 3581 -> 3581 | 0 -> 0 | dwordx4 32 | same |
| 256x256x64 | dp_ragged_wave | True | True | 512 -> 512 | 21 -> 21 | 3576 -> 3576 | 88 -> 88 | dwordx4 32 | same |
| 256x256x64 | kres_gt_dp | True | True | 456 -> 456 | 0 -> 0 | 3260 -> 3260 | 8 -> 8 | dwordx4 32 | same |
| 256x256x64 | kres_gt_rs | True | True | 492 -> 492 | 0 -> 0 | 4723 -> 4723 | 24 -> 24 | dwordx4 100 | same |
| 256x256x64 | kres_eq_dp | True | True | 444 -> 444 | 0 -> 0 | 3008 -> 3008 | 0 -> 0 | dwordx4 32 | same |
| 256x256x64 | kres_eq_owner | True | True | 512 -> 512 | 2 -> 2 | 7250 -> 7250 | 16 -> 16 | dwordx4 128 | same |
| 256x256x64 | dp_1wave | True | True | 464 -> 464 | 0 -> 0 | 3394 -> 3394 | 0 -> 0 | dwordx4 32 | same |
| 256x256x64 | dp_4wave | True | True | 464 -> 464 | 0 -> 0 | 3691 -> 3691 | 0 -> 0 | dwordx4 32 | same |
| 256x256x64 | dp_4w_kres | True | True | 452 -> 452 | 0 -> 0 | 3018 -> 3018 | 8 -> 8 | dwordx4 32 | same |
| 256x256x64 | rs_3wave | True | True | 484 -> 484 | 0 -> 0 | 4876 -> 4876 | 0 -> 0 | dwordx4 100 | same |
| 256x256x64 | rs_3w_kres | True | False | 492 -> 492 | 0 -> 0 | 5069 -> 5065 | 24 -> 24 | dwordx4 100 | same |
| 256x256x64 | l2_3w_kres | True | False | 480 -> 480 | 0 -> 0 | 4640 -> 4640 | 18 -> 18 | dwordx4 104, global_store_dword 2 | same |
| 256x256x64 | two_tile_2w | True | True | 512 -> 512 | 15 -> 15 | 12115 -> 12115 | 0 -> 0 | dwordx4 224 | same |
| 256x256x64 | two_tile_kres | True | False | 512 -> 512 | 35 -> 35 | 13064 -> 13064 | 140 -> 140 | dwordx4 224 | same |
| 256x256x64 | spread_2w | True | True | 512 -> 512 | 15 -> 15 | 12099 -> 12099 | 0 -> 0 | dwordx4 224 | same |
