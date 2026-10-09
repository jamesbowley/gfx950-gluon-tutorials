# check_streamk.py --policies capped (v20 WORKLOG entry 7), 30 rotating launches, after the n = 1 change

| shape | policy | fixup | tile order | bias | launches | wrong results | flags or counts left up |
|---|---|---|---|---|---|---|---|
| 1024x8192x3584 | capped | auto | v9 | 0 | 30 | 0 | 0 |
| 1024x8192x3584 | capped | auto | v9 | 1 | 30 | 0 | 0 |
| 1024x8192x3584 | capped | auto | split | 0 | 30 | 0 | 0 |
| 1024x8192x3584 | capped | auto | split | 1 | 30 | 0 | 0 |
| 4096x6144x3072 | capped | auto | v9 | 0 | 30 | 0 | 0 |
| 4096x6144x3072 | capped | auto | v9 | 1 | 30 | 0 | 0 |
| 4096x6144x3072 | capped | auto | split | 0 | 30 | 0 | 0 |
| 4096x6144x3072 | capped | auto | split | 1 | 30 | 0 | 0 |
| 5632x3072x3072 | capped | auto | v9 | 0 | 30 | 0 | 0 |
| 5632x3072x3072 | capped | auto | v9 | 1 | 30 | 0 | 0 |
| 5632x3072x3072 | capped | auto | split | 0 | 30 | 0 | 0 |
| 5632x3072x3072 | capped | auto | split | 1 | 30 | 0 | 0 |
| 4864x4096x3072 | capped | auto | v9 | 0 | 30 | 0 | 0 |
| 4864x4096x3072 | capped | auto | v9 | 1 | 30 | 0 | 0 |
| 4864x4096x3072 | capped | auto | split | 0 | 30 | 0 | 0 |
| 4864x4096x3072 | capped | auto | split | 1 | 30 | 0 | 0 |
| 512x4096x1536 | capped | auto | v9 | 0 | 30 | 0 | 0 |
| 512x4096x1536 | capped | auto | v9 | 1 | 30 | 0 | 0 |
| 512x4096x1536 | capped | auto | split | 0 | 30 | 0 | 0 |
| 512x4096x1536 | capped | auto | split | 1 | 30 | 0 | 0 |
| 18432x14592x8192 | capped | auto | v9 | 0 | 30 | 0 | 0 |
| 18432x14592x8192 | capped | auto | v9 | 1 | 30 | 0 | 0 |
| 18432x14592x8192 | capped | auto | split | 0 | 30 | 0 | 0 |
| 18432x14592x8192 | capped | auto | split | 1 | 30 | 0 | 0 |
| 12800x10752x6144 | capped | auto | v9 | 0 | 30 | 0 | 0 |
| 12800x10752x6144 | capped | auto | v9 | 1 | 30 | 0 | 0 |
| 12800x10752x6144 | capped | auto | split | 0 | 30 | 0 | 0 |
| 12800x10752x6144 | capped | auto | split | 1 | 30 | 0 | 0 |
| 8192x17152x6144 | capped | auto | v9 | 0 | 30 | 0 | 0 |
| 8192x17152x6144 | capped | auto | v9 | 1 | 30 | 0 | 0 |
| 8192x17152x6144 | capped | auto | split | 0 | 30 | 0 | 0 |
| 8192x17152x6144 | capped | auto | split | 1 | 30 | 0 | 0 |
| 2048x9216x16384 | capped | auto | v9 | 0 | 30 | 0 | 0 |
| 2048x9216x16384 | capped | auto | v9 | 1 | 30 | 0 | 0 |
| 2048x9216x16384 | capped | auto | split | 0 | 30 | 0 | 0 |
| 2048x9216x16384 | capped | auto | split | 1 | 30 | 0 | 0 |

PASS (36 configurations on GPUs 0-3)
