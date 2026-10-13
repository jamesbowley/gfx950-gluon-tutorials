| case | variant | correct | VGPR | VGPR spill | SGPR spill | scratch ops (stores / loads) | SGPR lanes | ending buffer ops |
|---|---|---|---|---|---|---|---|---|
| 4352x4096x8192:STREAMK_POLICY=two_tile | base | yes | 512 | 14 | 21 | L2.post 0/29, out 14/0 | 21 | load_dwordx4 224, store_dwordx4 128 |
| 4352x4096x8192:STREAMK_POLICY=two_tile | crow | yes | 512 | 5 | 25 | L2.post 0/9, out 5/0 | 25 | load_dwordx4 224, store_dwordx4 128 |
| 4352x4096x8192:STREAMK_POLICY=two_tile | ccol | yes | 512 | 69 | 25 | L2.post 16/28, out 5/0 | 25 | load_dwordx4 224, store_dwordx4 96, store_short 256 |
| 4352x4096x8192:STREAMK_POLICY=two_tile | cgrp | yes | 512 | 5 | 25 | L2.post 0/9, out 5/2 | 25 | load_dwordx4 224, store_dwordx4 128 |
| 4352x4096x8192:STREAMK_POLICY=two_tile | crow+pclamp | yes | 512 | 7 | 25 | L2.post 0/11, out 7/0 | 25 | load_dwordx4 224, store_dwordx4 128 |
| 4352x4096x8192:STREAMK_POLICY=two_tile | crow+pclamp2 | yes | 512 | 15 | 25 | L2.post 6/9, out 3/0 | 25 | load_dword 256, load_dwordx4 32, store_dword 256, store_dwordx4 64 |
| 4352x4096x8192:STREAMK_POLICY=two_tile | crow+plin | yes | 512 | 6 | 25 | L2.post 0/11, out 6/0 | 25 | load_dwordx2 384, load_dwordx4 32, store_dwordx2 128, store_dwordx4 64 |
| 4352x4096x8192:STREAMK_POLICY=two_tile | crow+bgrp | yes | 512 | 5 | 25 | L2.post 0/9, out 5/0 | 25 | load_dwordx4 224, store_dwordx4 128 |
| 4352x4096x8192:STREAMK_POLICY=two_tile | nofull | yes | 512 | 14 | 20 | L1.post 0/29, out 14/0 | 20 | load_dwordx4 192, store_dwordx4 96 |
| 4352x4096x8192:STREAMK_POLICY=two_tile | crow+nofull | yes | 512 | 7 | 28 | L1.post 0/22, out 7/0 | 28 | load_dwordx4 192, store_dwordx4 96 |
| 4352x4096x8192:STREAMK_POLICY=two_tile | split | yes | 512 | 13 | 32 | L2.post 0/13, out 13/0 | 32 | load_dwordx4 224, store_dwordx4 64 |
| 4352x4096x8192:STREAMK_POLICY=two_tile | crow+split | yes | 512 | 6 | 37 | L2.post 0/6, out 6/0 | 37 | load_dwordx4 224, store_dwordx4 64 |
| 4096x7168x32768:STREAMK_POLICY=spread | base | yes | 512 | 14 | 21 | L2.post 0/29, out 14/0 | 21 | load_dwordx4 224, store_dwordx4 128 |
| 4096x7168x32768:STREAMK_POLICY=spread | crow | yes | 512 | 5 | 25 | L2.post 0/9, out 5/0 | 25 | load_dwordx4 224, store_dwordx4 128 |
| 4096x7168x32768:STREAMK_POLICY=spread | ccol | yes | 512 | 69 | 25 | L2.post 16/28, out 5/0 | 25 | load_dwordx4 224, store_dwordx4 96, store_short 256 |
| 4096x7168x32768:STREAMK_POLICY=spread | cgrp | yes | 512 | 5 | 25 | L2.post 0/9, out 5/2 | 25 | load_dwordx4 224, store_dwordx4 128 |
| 4096x7168x32768:STREAMK_POLICY=spread | crow+pclamp | yes | 512 | 7 | 25 | L2.post 0/11, out 7/0 | 25 | load_dwordx4 224, store_dwordx4 128 |
| 4096x7168x32768:STREAMK_POLICY=spread | crow+pclamp2 | yes | 512 | 15 | 25 | L2.post 6/9, out 3/0 | 25 | load_dword 256, load_dwordx4 32, store_dword 256, store_dwordx4 64 |
| 4096x7168x32768:STREAMK_POLICY=spread | crow+plin | yes | 512 | 6 | 25 | L2.post 0/11, out 6/0 | 25 | load_dwordx2 384, load_dwordx4 32, store_dwordx2 128, store_dwordx4 64 |
| 4096x7168x32768:STREAMK_POLICY=spread | crow+bgrp | yes | 512 | 5 | 25 | L2.post 0/9, out 5/0 | 25 | load_dwordx4 224, store_dwordx4 128 |
| 4096x7168x32768:STREAMK_POLICY=spread | nofull | yes | 512 | 14 | 21 | L2.post 0/29, out 14/0 | 21 | load_dwordx4 224, store_dwordx4 128 |
| 4096x7168x32768:STREAMK_POLICY=spread | crow+nofull | yes | 512 | 5 | 25 | L2.post 0/9, out 5/0 | 25 | load_dwordx4 224, store_dwordx4 128 |
| 4096x7168x32768:STREAMK_POLICY=spread | split | yes | 512 | 13 | 32 | L2.post 0/13, out 13/0 | 32 | load_dwordx4 224, store_dwordx4 64 |
| 4096x7168x32768:STREAMK_POLICY=spread | crow+split | yes | 512 | 6 | 37 | L2.post 0/6, out 6/0 | 37 | load_dwordx4 224, store_dwordx4 64 |
| 4352x4096x8192:STREAMK_POLICY=two_tile:CASE_BIAS=1 | base | yes | 512 | 22 | 21 | L2.post 2/43, out 14/0 | 21 | load_dwordx2 16, load_dwordx4 224, store_dwordx4 128 |
| 4352x4096x8192:STREAMK_POLICY=two_tile:CASE_BIAS=1 | crow | yes | 512 | 31 | 27 | L2.post 6/34, out 7/0 | 27 | load_dwordx2 16, load_dwordx4 224, store_dwordx4 128 |
| 4352x4096x8192:STREAMK_POLICY=two_tile:CASE_BIAS=1 | ccol | yes | 512 | 93 | 27 | L2.post 22/39, out 5/0 | 27 | load_dwordx2 16, load_dwordx4 224, store_dwordx4 96, store_short 256 |
| 4352x4096x8192:STREAMK_POLICY=two_tile:CASE_BIAS=1 | cgrp | yes | 512 | 32 | 27 | L2.post 6/35, out 8/1 | 27 | load_dwordx2 16, load_dwordx4 224, store_dwordx4 128 |
| 4352x4096x8192:STREAMK_POLICY=two_tile:CASE_BIAS=1 | crow+pclamp | yes | 512 | 36 | 27 | L2.post 6/45, out 12/0 | 27 | load_dwordx2 16, load_dwordx4 224, store_dwordx4 128 |
| 4352x4096x8192:STREAMK_POLICY=two_tile:CASE_BIAS=1 | crow+pclamp2 | yes | 512 | 35 | 28 | L2.post 14/21, out 7/0 | 28 | load_dword 256, load_dwordx2 16, load_dwordx4 32, store_dword 256, store_dwordx4 64 |
| 4352x4096x8192:STREAMK_POLICY=two_tile:CASE_BIAS=1 | crow+plin | yes | 512 | 22 | 27 | L2.post 6/25, out 10/0 | 27 | load_dwordx2 400, load_dwordx4 32, store_dwordx2 128, store_dwordx4 64 |
| 4352x4096x8192:STREAMK_POLICY=two_tile:CASE_BIAS=1 | crow+bgrp | yes | 512 | 31 | 28 | L2.post 6/34, out 7/0 | 28 | load_dwordx2 16, load_dwordx4 224, store_dwordx4 128 |
| 4352x4096x8192:STREAMK_POLICY=two_tile:CASE_BIAS=1 | nofull | yes | 512 | 26 | 24 | L1.post 2/47, out 18/0 | 24 | load_dwordx2 16, load_dwordx4 192, store_dwordx4 96 |
| 4352x4096x8192:STREAMK_POLICY=two_tile:CASE_BIAS=1 | crow+nofull | yes | 512 | 35 | 30 | L1.post 6/36, out 11/0 | 30 | load_dwordx2 16, load_dwordx4 192, store_dwordx4 96 |
| 4352x4096x8192:STREAMK_POLICY=two_tile:CASE_BIAS=1 | split | yes | 512 | 33 | 36 | L2.post 2/26, L2.pre 0/16, out 29/0 | 36 | load_dwordx2 16, load_dwordx4 224, store_dwordx4 64 |
| 4352x4096x8192:STREAMK_POLICY=two_tile:CASE_BIAS=1 | crow+split | yes | 512 | 26 | 42 | L2.post 2/19, L2.pre 0/12, out 22/0 | 42 | load_dwordx2 16, load_dwordx4 224, store_dwordx4 64 |
