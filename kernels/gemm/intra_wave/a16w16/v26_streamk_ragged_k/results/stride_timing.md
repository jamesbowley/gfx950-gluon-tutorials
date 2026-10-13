# Cold kernel-only times with an explicit row stride (stride_rp.py: rocprofv3 --kernel-trace over stride_bench.py, 300 launches over rotating copies > 512 MB, median), GPU 1

## v9_ragged, residue first with the whole loop shifted by K_RES (first attempt, replaced)

| version | shape | row stride | env | cold kernel us |
|---|---|---|---|---|
| 9 | 4096x4096x8192 | 8256 | - | 183.7 |
| 9r | 4096x4096x8192 | 8256 | - | 183.5 |
| 9r | 4096x4096x8176 | 8256 | - | 287.9 |
| 9r | 4096x4096x8128 | 8256 | - | 181.7 |
| 9r | 4096x4096x8080 | 8256 | - | 283.8 |
| 9r | 4096x4096x8192 | 8192 | - | 180.6 |
| 9r | 4096x4096x8176 | 8176 | - | 220.3 |
| 9r | 4096x4096x8128 | 8128 | - | 182.6 |
| 9r | 4096x4096x8080 | 8080 | - | 224.7 |

## v9_ragged, last pair first (as committed)

| version | shape | row stride | env | cold kernel us |
|---|---|---|---|---|
| 9r | 4096x4096x8192 | 8256 | - | 183.2 |
| 9r | 4096x4096x8176 | 8256 | - | 182.4 |
| 9r | 4096x4096x8128 | 8256 | - | 183.4 |
| 9r | 4096x4096x8080 | 8256 | - | 183.3 |
| 9r | 4096x4096x8176 | 8176 | - | 221.4 |
| 9r | 4096x4096x8080 | 8080 | - | 224.1 |

## v9 (unchanged) at K = 8192 with a misaligned row stride

| version | shape | row stride | env | cold kernel us |
|---|---|---|---|---|
| 9 | 4096x4096x8192 | 8208 | - | 220.7 |
| 9 | 4096x4096x8192 | 8256 | - | 184.7 |

## v9_tile (256x256)

| version | shape | row stride | env | cold kernel us |
|---|---|---|---|---|
| 9t | 4096x4096x8192 | 8256 | - | 183.7 |
| 9t | 4096x4096x8176 | 8256 | - | 182.9 |
| 9t | 4096x4096x8080 | 8256 | - | 184.1 |
| 9t | 4000x4000x8080 | 8256 | - | 177.7 |

## v25 / v26

| version | shape | row stride | env | cold kernel us |
|---|---|---|---|---|
| 25 | 4352x4096x8192 | 8256 | - | 223.7 |
| 26 | 4352x4096x8192 | 8256 | - | 224.4 |
| 26 | 4352x4096x8176 | 8256 | - | 220.6 |
| 26 | 4352x4096x8128 | 8256 | - | 220.4 |
| 26 | 4352x4096x8080 | 8256 | - | 220.0 |
| 25 | 8192x8448x8192 | 8256 | - | 735.2 |
| 26 | 8192x8448x8176 | 8256 | - | 734.3 |
| 26 | 8192x8448x8080 | 8256 | - | 735.2 |
| 25 | 8192x8192x8192 | 8256 | - | 692.1 |
| 26 | 8192x8192x8176 | 8256 | - | 690.3 |
| 26 | 8192x8192x8080 | 8256 | - | 687.0 |
| 25 | 4352x4096x8192 | 8256 | STREAMK_FORCE_EDGE=1 | 223.6 |
| 26 | 4300x4000x8176 | 8256 | - | 216.7 |
| 26 | 4300x4000x8080 | 8256 | - | 214.6 |
| 25 | 4096x6144x8192 | 8256 | - | 277.4 |
| 26 | 4096x6144x8080 | 8256 | - | 276.8 |
