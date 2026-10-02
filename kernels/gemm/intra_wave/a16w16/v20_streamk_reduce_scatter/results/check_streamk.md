# check_streamk.py for v20 (30 rotating launches; the capped shapes 10), HIP_VISIBLE_DEVICES=3

| shape | policy | fixup | tile order | bias | launches | wrong results | flags or counts left up |
|---|---|---|---|---|---|---|---|
| 4352x4096x8192 | one_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 4352x4096x8192 | one_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 4352x4096x8192 | one_tile | rs | split | 0 | 30 | 0 | 0 |
| 4352x4096x8192 | one_tile | rs | split | 1 | 30 | 0 | 0 |
| 4352x4096x8192 | one_tile | owner | v9 | 0 | 30 | 0 | 0 |
| 4352x4096x8192 | one_tile | owner | v9 | 1 | 30 | 0 | 0 |
| 4352x4096x8192 | one_tile | owner | split | 0 | 30 | 0 | 0 |
| 4352x4096x8192 | one_tile | owner | split | 1 | 30 | 0 | 0 |
| 4352x4096x8192 | two_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 4352x4096x8192 | two_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 4352x4096x8192 | two_tile | rs | split | 0 | 30 | 0 | 0 |
| 4352x4096x8192 | two_tile | rs | split | 1 | 30 | 0 | 0 |
| 4352x4352x8192 | one_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 4352x4352x8192 | one_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 4352x4352x8192 | one_tile | rs | split | 0 | 30 | 0 | 0 |
| 4352x4352x8192 | one_tile | rs | split | 1 | 30 | 0 | 0 |
| 4352x4352x8192 | one_tile | owner | v9 | 0 | 30 | 0 | 0 |
| 4352x4352x8192 | one_tile | owner | v9 | 1 | 30 | 0 | 0 |
| 4352x4352x8192 | one_tile | owner | split | 0 | 30 | 0 | 0 |
| 4352x4352x8192 | one_tile | owner | split | 1 | 30 | 0 | 0 |
| 4352x4352x8192 | two_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 4352x4352x8192 | two_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 4352x4352x8192 | two_tile | rs | split | 0 | 30 | 0 | 0 |
| 4352x4352x8192 | two_tile | rs | split | 1 | 30 | 0 | 0 |
| 4352x4096x4096 | one_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 4352x4096x4096 | one_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 4352x4096x4096 | one_tile | rs | split | 0 | 30 | 0 | 0 |
| 4352x4096x4096 | one_tile | rs | split | 1 | 30 | 0 | 0 |
| 4352x4096x4096 | one_tile | owner | v9 | 0 | 30 | 0 | 0 |
| 4352x4096x4096 | one_tile | owner | v9 | 1 | 30 | 0 | 0 |
| 4352x4096x4096 | one_tile | owner | split | 0 | 30 | 0 | 0 |
| 4352x4096x4096 | one_tile | owner | split | 1 | 30 | 0 | 0 |
| 4352x4096x4096 | two_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 4352x4096x4096 | two_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 4352x4096x4096 | two_tile | rs | split | 0 | 30 | 0 | 0 |
| 4352x4096x4096 | two_tile | rs | split | 1 | 30 | 0 | 0 |
| 4352x4096x16384 | one_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 4352x4096x16384 | one_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 4352x4096x16384 | one_tile | rs | split | 0 | 30 | 0 | 0 |
| 4352x4096x16384 | one_tile | rs | split | 1 | 30 | 0 | 0 |
| 4352x4096x16384 | one_tile | owner | v9 | 0 | 30 | 0 | 0 |
| 4352x4096x16384 | one_tile | owner | v9 | 1 | 30 | 0 | 0 |
| 4352x4096x16384 | one_tile | owner | split | 0 | 30 | 0 | 0 |
| 4352x4096x16384 | one_tile | owner | split | 1 | 30 | 0 | 0 |
| 4352x4096x16384 | two_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 4352x4096x16384 | two_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 4352x4096x16384 | two_tile | rs | split | 0 | 30 | 0 | 0 |
| 4352x4096x16384 | two_tile | rs | split | 1 | 30 | 0 | 0 |
| 3328x5120x8192 | one_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 3328x5120x8192 | one_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 3328x5120x8192 | one_tile | rs | split | 0 | 30 | 0 | 0 |
| 3328x5120x8192 | one_tile | rs | split | 1 | 30 | 0 | 0 |
| 3328x5120x8192 | one_tile | owner | v9 | 0 | 30 | 0 | 0 |
| 3328x5120x8192 | one_tile | owner | v9 | 1 | 30 | 0 | 0 |
| 3328x5120x8192 | one_tile | owner | split | 0 | 30 | 0 | 0 |
| 3328x5120x8192 | one_tile | owner | split | 1 | 30 | 0 | 0 |
| 3328x5120x8192 | two_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 3328x5120x8192 | two_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 3328x5120x8192 | two_tile | rs | split | 0 | 30 | 0 | 0 |
| 3328x5120x8192 | two_tile | rs | split | 1 | 30 | 0 | 0 |
| 4096x6144x8192 | one_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 4096x6144x8192 | one_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 4096x6144x8192 | one_tile | rs | split | 0 | 30 | 0 | 0 |
| 4096x6144x8192 | one_tile | rs | split | 1 | 30 | 0 | 0 |
| 4096x6144x8192 | one_tile | owner | v9 | 0 | 30 | 0 | 0 |
| 4096x6144x8192 | one_tile | owner | v9 | 1 | 30 | 0 | 0 |
| 4096x6144x8192 | one_tile | owner | split | 0 | 30 | 0 | 0 |
| 4096x6144x8192 | one_tile | owner | split | 1 | 30 | 0 | 0 |
| 4096x6144x8192 | two_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 4096x6144x8192 | two_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 4096x6144x8192 | two_tile | rs | split | 0 | 30 | 0 | 0 |
| 4096x6144x8192 | two_tile | rs | split | 1 | 30 | 0 | 0 |
| 3840x4096x8192 | one_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 3840x4096x8192 | one_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 3840x4096x8192 | one_tile | rs | split | 0 | 30 | 0 | 0 |
| 3840x4096x8192 | one_tile | rs | split | 1 | 30 | 0 | 0 |
| 3840x4096x8192 | one_tile | owner | v9 | 0 | 30 | 0 | 0 |
| 3840x4096x8192 | one_tile | owner | v9 | 1 | 30 | 0 | 0 |
| 3840x4096x8192 | one_tile | owner | split | 0 | 30 | 0 | 0 |
| 3840x4096x8192 | one_tile | owner | split | 1 | 30 | 0 | 0 |
| 3840x4096x8192 | two_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 3840x4096x8192 | two_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 3840x4096x8192 | two_tile | rs | split | 0 | 30 | 0 | 0 |
| 3840x4096x8192 | two_tile | rs | split | 1 | 30 | 0 | 0 |
| 8192x8448x8192 | one_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 8192x8448x8192 | one_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 8192x8448x8192 | one_tile | rs | split | 0 | 30 | 0 | 0 |
| 8192x8448x8192 | one_tile | rs | split | 1 | 30 | 0 | 0 |
| 8192x8448x8192 | one_tile | owner | v9 | 0 | 30 | 0 | 0 |
| 8192x8448x8192 | one_tile | owner | v9 | 1 | 30 | 0 | 0 |
| 8192x8448x8192 | one_tile | owner | split | 0 | 30 | 0 | 0 |
| 8192x8448x8192 | one_tile | owner | split | 1 | 30 | 0 | 0 |
| 8192x8448x8192 | two_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 8192x8448x8192 | two_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 8192x8448x8192 | two_tile | rs | split | 0 | 30 | 0 | 0 |
| 8192x8448x8192 | two_tile | rs | split | 1 | 30 | 0 | 0 |
| 8192x7936x8192 | one_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 8192x7936x8192 | one_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 8192x7936x8192 | one_tile | rs | split | 0 | 30 | 0 | 0 |
| 8192x7936x8192 | one_tile | rs | split | 1 | 30 | 0 | 0 |
| 8192x7936x8192 | one_tile | owner | v9 | 0 | 30 | 0 | 0 |
| 8192x7936x8192 | one_tile | owner | v9 | 1 | 30 | 0 | 0 |
| 8192x7936x8192 | one_tile | owner | split | 0 | 30 | 0 | 0 |
| 8192x7936x8192 | one_tile | owner | split | 1 | 30 | 0 | 0 |
| 8192x7936x8192 | two_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 8192x7936x8192 | two_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 8192x7936x8192 | two_tile | rs | split | 0 | 30 | 0 | 0 |
| 8192x7936x8192 | two_tile | rs | split | 1 | 30 | 0 | 0 |
PASS

| shape | policy | fixup | tile order | bias | launches | wrong results | flags or counts left up |
|---|---|---|---|---|---|---|---|
| 1536x11008x8192 | one_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 1536x11008x8192 | one_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 1536x11008x8192 | one_tile | rs | split | 0 | 30 | 0 | 0 |
| 1536x11008x8192 | one_tile | rs | split | 1 | 30 | 0 | 0 |
| 1536x11008x8192 | one_tile | owner | v9 | 0 | 30 | 0 | 0 |
| 1536x11008x8192 | one_tile | owner | v9 | 1 | 30 | 0 | 0 |
| 1536x11008x8192 | one_tile | owner | split | 0 | 30 | 0 | 0 |
| 1536x11008x8192 | one_tile | owner | split | 1 | 30 | 0 | 0 |
| 1536x11008x8192 | two_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 1536x11008x8192 | two_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 1536x11008x8192 | two_tile | rs | split | 0 | 30 | 0 | 0 |
| 1536x11008x8192 | two_tile | rs | split | 1 | 30 | 0 | 0 |
| 4096x5120x8192 | one_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 4096x5120x8192 | one_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 4096x5120x8192 | one_tile | rs | split | 0 | 30 | 0 | 0 |
| 4096x5120x8192 | one_tile | rs | split | 1 | 30 | 0 | 0 |
| 4096x5120x8192 | one_tile | owner | v9 | 0 | 30 | 0 | 0 |
| 4096x5120x8192 | one_tile | owner | v9 | 1 | 30 | 0 | 0 |
| 4096x5120x8192 | one_tile | owner | split | 0 | 30 | 0 | 0 |
| 4096x5120x8192 | one_tile | owner | split | 1 | 30 | 0 | 0 |
| 4096x5120x8192 | two_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 4096x5120x8192 | two_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 4096x5120x8192 | two_tile | rs | split | 0 | 30 | 0 | 0 |
| 4096x5120x8192 | two_tile | rs | split | 1 | 30 | 0 | 0 |
| 4096x7168x4096 | one_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 4096x7168x4096 | one_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 4096x7168x4096 | one_tile | rs | split | 0 | 30 | 0 | 0 |
| 4096x7168x4096 | one_tile | rs | split | 1 | 30 | 0 | 0 |
| 4096x7168x4096 | one_tile | owner | v9 | 0 | 30 | 0 | 0 |
| 4096x7168x4096 | one_tile | owner | v9 | 1 | 30 | 0 | 0 |
| 4096x7168x4096 | one_tile | owner | split | 0 | 30 | 0 | 0 |
| 4096x7168x4096 | one_tile | owner | split | 1 | 30 | 0 | 0 |
| 4096x7168x4096 | two_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 4096x7168x4096 | two_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 4096x7168x4096 | two_tile | rs | split | 0 | 30 | 0 | 0 |
| 4096x7168x4096 | two_tile | rs | split | 1 | 30 | 0 | 0 |
| 4096x7168x8192 | one_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 4096x7168x8192 | one_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 4096x7168x8192 | one_tile | rs | split | 0 | 30 | 0 | 0 |
| 4096x7168x8192 | one_tile | rs | split | 1 | 30 | 0 | 0 |
| 4096x7168x8192 | one_tile | owner | v9 | 0 | 30 | 0 | 0 |
| 4096x7168x8192 | one_tile | owner | v9 | 1 | 30 | 0 | 0 |
| 4096x7168x8192 | one_tile | owner | split | 0 | 30 | 0 | 0 |
| 4096x7168x8192 | one_tile | owner | split | 1 | 30 | 0 | 0 |
| 4096x7168x8192 | two_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 4096x7168x8192 | two_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 4096x7168x8192 | two_tile | rs | split | 0 | 30 | 0 | 0 |
| 4096x7168x8192 | two_tile | rs | split | 1 | 30 | 0 | 0 |
| 4096x7168x16384 | one_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 4096x7168x16384 | one_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 4096x7168x16384 | one_tile | rs | split | 0 | 30 | 0 | 0 |
| 4096x7168x16384 | one_tile | rs | split | 1 | 30 | 0 | 0 |
| 4096x7168x16384 | one_tile | owner | v9 | 0 | 30 | 0 | 0 |
| 4096x7168x16384 | one_tile | owner | v9 | 1 | 30 | 0 | 0 |
| 4096x7168x16384 | one_tile | owner | split | 0 | 30 | 0 | 0 |
| 4096x7168x16384 | one_tile | owner | split | 1 | 30 | 0 | 0 |
| 4096x7168x16384 | two_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 4096x7168x16384 | two_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 4096x7168x16384 | two_tile | rs | split | 0 | 30 | 0 | 0 |
| 4096x7168x16384 | two_tile | rs | split | 1 | 30 | 0 | 0 |
| 9472x10240x8192 | one_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 9472x10240x8192 | one_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 9472x10240x8192 | one_tile | rs | split | 0 | 30 | 0 | 0 |
| 9472x10240x8192 | one_tile | rs | split | 1 | 30 | 0 | 0 |
| 9472x10240x8192 | one_tile | owner | v9 | 0 | 30 | 0 | 0 |
| 9472x10240x8192 | one_tile | owner | v9 | 1 | 30 | 0 | 0 |
| 9472x10240x8192 | one_tile | owner | split | 0 | 30 | 0 | 0 |
| 9472x10240x8192 | one_tile | owner | split | 1 | 30 | 0 | 0 |
| 9472x10240x8192 | two_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 9472x10240x8192 | two_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 9472x10240x8192 | two_tile | rs | split | 0 | 30 | 0 | 0 |
| 9472x10240x8192 | two_tile | rs | split | 1 | 30 | 0 | 0 |
| 4096x7936x8192 | one_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 4096x7936x8192 | one_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 4096x7936x8192 | one_tile | rs | split | 0 | 30 | 0 | 0 |
| 4096x7936x8192 | one_tile | rs | split | 1 | 30 | 0 | 0 |
| 4096x7936x8192 | one_tile | owner | v9 | 0 | 30 | 0 | 0 |
| 4096x7936x8192 | one_tile | owner | v9 | 1 | 30 | 0 | 0 |
| 4096x7936x8192 | one_tile | owner | split | 0 | 30 | 0 | 0 |
| 4096x7936x8192 | one_tile | owner | split | 1 | 30 | 0 | 0 |
| 4096x7936x8192 | two_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 4096x7936x8192 | two_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 4096x7936x8192 | two_tile | rs | split | 0 | 30 | 0 | 0 |
| 4096x7936x8192 | two_tile | rs | split | 1 | 30 | 0 | 0 |
| 4608x7168x8192 | one_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 4608x7168x8192 | one_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 4608x7168x8192 | one_tile | rs | split | 0 | 30 | 0 | 0 |
| 4608x7168x8192 | one_tile | rs | split | 1 | 30 | 0 | 0 |
| 4608x7168x8192 | one_tile | owner | v9 | 0 | 30 | 0 | 0 |
| 4608x7168x8192 | one_tile | owner | v9 | 1 | 30 | 0 | 0 |
| 4608x7168x8192 | one_tile | owner | split | 0 | 30 | 0 | 0 |
| 4608x7168x8192 | one_tile | owner | split | 1 | 30 | 0 | 0 |
| 4608x7168x8192 | two_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 4608x7168x8192 | two_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 4608x7168x8192 | two_tile | rs | split | 0 | 30 | 0 | 0 |
| 4608x7168x8192 | two_tile | rs | split | 1 | 30 | 0 | 0 |
| 1792x18688x8192 | one_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 1792x18688x8192 | one_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 1792x18688x8192 | one_tile | rs | split | 0 | 30 | 0 | 0 |
| 1792x18688x8192 | one_tile | rs | split | 1 | 30 | 0 | 0 |
| 1792x18688x8192 | one_tile | owner | v9 | 0 | 30 | 0 | 0 |
| 1792x18688x8192 | one_tile | owner | v9 | 1 | 30 | 0 | 0 |
| 1792x18688x8192 | one_tile | owner | split | 0 | 30 | 0 | 0 |
| 1792x18688x8192 | one_tile | owner | split | 1 | 30 | 0 | 0 |
| 1792x18688x8192 | two_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 1792x18688x8192 | two_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 1792x18688x8192 | two_tile | rs | split | 0 | 30 | 0 | 0 |
| 1792x18688x8192 | two_tile | rs | split | 1 | 30 | 0 | 0 |
| 1536x11008x16384 | one_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 1536x11008x16384 | one_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 1536x11008x16384 | one_tile | rs | split | 0 | 30 | 0 | 0 |
| 1536x11008x16384 | one_tile | rs | split | 1 | 30 | 0 | 0 |
| 1536x11008x16384 | one_tile | owner | v9 | 0 | 30 | 0 | 0 |
| 1536x11008x16384 | one_tile | owner | v9 | 1 | 30 | 0 | 0 |
| 1536x11008x16384 | one_tile | owner | split | 0 | 30 | 0 | 0 |
| 1536x11008x16384 | one_tile | owner | split | 1 | 30 | 0 | 0 |
| 1536x11008x16384 | two_tile | rs | v9 | 0 | 30 | 0 | 0 |
| 1536x11008x16384 | two_tile | rs | v9 | 1 | 30 | 0 | 0 |
| 1536x11008x16384 | two_tile | rs | split | 0 | 30 | 0 | 0 |
| 1536x11008x16384 | two_tile | rs | split | 1 | 30 | 0 | 0 |
PASS

| shape | policy | fixup | tile order | bias | launches | wrong results | flags or counts left up |
|---|---|---|---|---|---|---|---|
| 256x65792x8192 | one_tile | rs | v9 | 0 | 10 | 0 | 0 |
| 256x65792x8192 | one_tile | rs | v9 | 1 | 10 | 0 | 0 |
| 256x65792x8192 | one_tile | rs | split | 0 | 10 | 0 | 0 |
| 256x65792x8192 | one_tile | rs | split | 1 | 10 | 0 | 0 |
| 256x65792x8192 | one_tile | owner | v9 | 0 | 10 | 0 | 0 |
| 256x65792x8192 | one_tile | owner | v9 | 1 | 10 | 0 | 0 |
| 256x65792x8192 | one_tile | owner | split | 0 | 10 | 0 | 0 |
| 256x65792x8192 | one_tile | owner | split | 1 | 10 | 0 | 0 |
| 256x67328x4096 | one_tile | rs | v9 | 0 | 10 | 0 | 0 |
| 256x67328x4096 | one_tile | rs | v9 | 1 | 10 | 0 | 0 |
| 256x67328x4096 | one_tile | rs | split | 0 | 10 | 0 | 0 |
| 256x67328x4096 | one_tile | rs | split | 1 | 10 | 0 | 0 |
| 256x67328x4096 | one_tile | owner | v9 | 0 | 10 | 0 | 0 |
| 256x67328x4096 | one_tile | owner | v9 | 1 | 10 | 0 | 0 |
| 256x67328x4096 | one_tile | owner | split | 0 | 10 | 0 | 0 |
| 256x67328x4096 | one_tile | owner | split | 1 | 10 | 0 | 0 |
PASS

## Negative test (--stale-count: tile 0's barrier count preset to n - 1, 10 launches, 120 s timeout)

| shape | policy | fixup | tile order | bias | launches | wrong results | flags or counts left up |
|---|---|---|---|---|---|---|---|
| 4352x4096x8192 | one_tile | rs | v9 | 0 | 10 | error: no result after 120 s (hang) | |
| 4352x4352x8192 | one_tile | rs | v9 | 0 | 10 | 0 | 1 |

FAIL (as it must)

## After the host rule (entry 2): STREAMK_POLICY=auto STREAMK_FIXUP=auto, 30 rotating launches

| shape | policy | fixup | tile order | bias | launches | wrong results | flags or counts left up |
|---|---|---|---|---|---|---|---|
| 4352x4096x8192 | auto | auto | v9 | 0 | 30 | 0 | 0 |
| 4352x4096x8192 | auto | auto | v9 | 1 | 30 | 0 | 0 |
| 4352x4096x8192 | auto | auto | split | 0 | 30 | 0 | 0 |
| 4352x4096x8192 | auto | auto | split | 1 | 30 | 0 | 0 |
| 4352x4352x8192 | auto | auto | v9 | 0 | 30 | 0 | 0 |
| 4352x4352x8192 | auto | auto | v9 | 1 | 30 | 0 | 0 |
| 4352x4352x8192 | auto | auto | split | 0 | 30 | 0 | 0 |
| 4352x4352x8192 | auto | auto | split | 1 | 30 | 0 | 0 |
| 4352x4096x4096 | auto | auto | v9 | 0 | 30 | 0 | 0 |
| 4352x4096x4096 | auto | auto | v9 | 1 | 30 | 0 | 0 |
| 4352x4096x4096 | auto | auto | split | 0 | 30 | 0 | 0 |
| 4352x4096x4096 | auto | auto | split | 1 | 30 | 0 | 0 |
| 4352x4096x16384 | auto | auto | v9 | 0 | 30 | 0 | 0 |
| 4352x4096x16384 | auto | auto | v9 | 1 | 30 | 0 | 0 |
| 4352x4096x16384 | auto | auto | split | 0 | 30 | 0 | 0 |
| 4352x4096x16384 | auto | auto | split | 1 | 30 | 0 | 0 |
| 3328x5120x8192 | auto | auto | v9 | 0 | 30 | 0 | 0 |
| 3328x5120x8192 | auto | auto | v9 | 1 | 30 | 0 | 0 |
| 3328x5120x8192 | auto | auto | split | 0 | 30 | 0 | 0 |
| 3328x5120x8192 | auto | auto | split | 1 | 30 | 0 | 0 |
| 4096x6144x8192 | auto | auto | v9 | 0 | 30 | 0 | 0 |
| 4096x6144x8192 | auto | auto | v9 | 1 | 30 | 0 | 0 |
| 4096x6144x8192 | auto | auto | split | 0 | 30 | 0 | 0 |
| 4096x6144x8192 | auto | auto | split | 1 | 30 | 0 | 0 |
| 3840x4096x8192 | auto | auto | v9 | 0 | 30 | 0 | 0 |
| 3840x4096x8192 | auto | auto | v9 | 1 | 30 | 0 | 0 |
| 3840x4096x8192 | auto | auto | split | 0 | 30 | 0 | 0 |
| 3840x4096x8192 | auto | auto | split | 1 | 30 | 0 | 0 |
| 8192x8448x8192 | auto | auto | v9 | 0 | 30 | 0 | 0 |
| 8192x8448x8192 | auto | auto | v9 | 1 | 30 | 0 | 0 |
| 8192x8448x8192 | auto | auto | split | 0 | 30 | 0 | 0 |
| 8192x8448x8192 | auto | auto | split | 1 | 30 | 0 | 0 |
| 8192x7936x8192 | auto | auto | v9 | 0 | 30 | 0 | 0 |
| 8192x7936x8192 | auto | auto | v9 | 1 | 30 | 0 | 0 |
| 8192x7936x8192 | auto | auto | split | 0 | 30 | 0 | 0 |
| 8192x7936x8192 | auto | auto | split | 1 | 30 | 0 | 0 |
| 1536x11008x8192 | auto | auto | v9 | 0 | 30 | 0 | 0 |
| 1536x11008x8192 | auto | auto | v9 | 1 | 30 | 0 | 0 |
| 1536x11008x8192 | auto | auto | split | 0 | 30 | 0 | 0 |
| 1536x11008x8192 | auto | auto | split | 1 | 30 | 0 | 0 |
| 4096x5120x8192 | auto | auto | v9 | 0 | 30 | 0 | 0 |
| 4096x5120x8192 | auto | auto | v9 | 1 | 30 | 0 | 0 |
| 4096x5120x8192 | auto | auto | split | 0 | 30 | 0 | 0 |
| 4096x5120x8192 | auto | auto | split | 1 | 30 | 0 | 0 |
| 4096x7168x4096 | auto | auto | v9 | 0 | 30 | 0 | 0 |
| 4096x7168x4096 | auto | auto | v9 | 1 | 30 | 0 | 0 |
| 4096x7168x4096 | auto | auto | split | 0 | 30 | 0 | 0 |
| 4096x7168x4096 | auto | auto | split | 1 | 30 | 0 | 0 |
| 4096x7936x8192 | auto | auto | v9 | 0 | 30 | 0 | 0 |
| 4096x7936x8192 | auto | auto | v9 | 1 | 30 | 0 | 0 |
| 4096x7936x8192 | auto | auto | split | 0 | 30 | 0 | 0 |
| 4096x7936x8192 | auto | auto | split | 1 | 30 | 0 | 0 |
| 1792x18688x8192 | auto | auto | v9 | 0 | 30 | 0 | 0 |
| 1792x18688x8192 | auto | auto | v9 | 1 | 30 | 0 | 0 |
| 1792x18688x8192 | auto | auto | split | 0 | 30 | 0 | 0 |
| 1792x18688x8192 | auto | auto | split | 1 | 30 | 0 | 0 |
| 4096x4608x16384 | auto | auto | v9 | 0 | 30 | 0 | 0 |
| 4096x4608x16384 | auto | auto | v9 | 1 | 30 | 0 | 0 |
| 4096x4608x16384 | auto | auto | split | 0 | 30 | 0 | 0 |
| 4096x4608x16384 | auto | auto | split | 1 | 30 | 0 | 0 |
| 4352x4352x16384 | auto | auto | v9 | 0 | 30 | 0 | 0 |
| 4352x4352x16384 | auto | auto | v9 | 1 | 30 | 0 | 0 |
| 4352x4352x16384 | auto | auto | split | 0 | 30 | 0 | 0 |
| 4352x4352x16384 | auto | auto | split | 1 | 30 | 0 | 0 |
| 256x65792x8192 | auto | auto | v9 | 0 | 30 | 0 | 0 |
| 256x65792x8192 | auto | auto | v9 | 1 | 30 | 0 | 0 |
| 256x65792x8192 | auto | auto | split | 0 | 30 | 0 | 0 |
| 256x65792x8192 | auto | auto | split | 1 | 30 | 0 | 0 |
PASS

## STREAMK_POLICY=spread (end-to-end reversed split on the one-tile tiles, WORKLOG entry 3), 30 rotating launches

| shape | policy | fixup | tile order | bias | launches | wrong results | flags or counts left up |
|---|---|---|---|---|---|---|---|
| 4096x7168x8192 | spread | rs | v9 | 0 | 30 | 0 | 0 |
| 4096x7168x8192 | spread | rs | v9 | 1 | 30 | 0 | 0 |
| 4096x7168x8192 | spread | rs | split | 0 | 30 | 0 | 0 |
| 4096x7168x8192 | spread | rs | split | 1 | 30 | 0 | 0 |
| 4096x7168x16384 | spread | rs | v9 | 0 | 30 | 0 | 0 |
| 4096x7168x16384 | spread | rs | v9 | 1 | 30 | 0 | 0 |
| 4096x7168x16384 | spread | rs | split | 0 | 30 | 0 | 0 |
| 4096x7168x16384 | spread | rs | split | 1 | 30 | 0 | 0 |
| 4096x7168x32768 | spread | rs | v9 | 0 | 30 | 0 | 0 |
| 4096x7168x32768 | spread | rs | v9 | 1 | 30 | 0 | 0 |
| 4096x7168x32768 | spread | rs | split | 0 | 30 | 0 | 0 |
| 4096x7168x32768 | spread | rs | split | 1 | 30 | 0 | 0 |
| 4096x7168x65536 | spread | rs | v9 | 0 | 30 | 0 | 0 |
| 4096x7168x65536 | spread | rs | v9 | 1 | 30 | 0 | 0 |
| 4096x7168x65536 | spread | rs | split | 0 | 30 | 0 | 0 |
| 4096x7168x65536 | spread | rs | split | 1 | 30 | 0 | 0 |
| 8192x7936x32768 | spread | rs | v9 | 0 | 30 | 0 | 0 |
| 8192x7936x32768 | spread | rs | v9 | 1 | 30 | 0 | 0 |
| 8192x7936x32768 | spread | rs | split | 0 | 30 | 0 | 0 |
| 8192x7936x32768 | spread | rs | split | 1 | 30 | 0 | 0 |
| 4096x7936x32768 | spread | rs | v9 | 0 | 30 | 0 | 0 |
| 4096x7936x32768 | spread | rs | v9 | 1 | 30 | 0 | 0 |
| 4096x7936x32768 | spread | rs | split | 0 | 30 | 0 | 0 |
| 4096x7936x32768 | spread | rs | split | 1 | 30 | 0 | 0 |
| 4096x6144x8192 | spread | rs | v9 | 0 | 30 | 0 | 0 |
| 4096x6144x8192 | spread | rs | v9 | 1 | 30 | 0 | 0 |
| 4096x6144x8192 | spread | rs | split | 0 | 30 | 0 | 0 |
| 4096x6144x8192 | spread | rs | split | 1 | 30 | 0 | 0 |
| 3328x5120x8192 | spread | rs | v9 | 0 | 30 | 0 | 0 |
| 3328x5120x8192 | spread | rs | v9 | 1 | 30 | 0 | 0 |
| 3328x5120x8192 | spread | rs | split | 0 | 30 | 0 | 0 |
| 3328x5120x8192 | spread | rs | split | 1 | 30 | 0 | 0 |
| 4352x4096x8192 | spread | rs | v9 | 0 | 30 | 0 | 0 |
| 4352x4096x8192 | spread | rs | v9 | 1 | 30 | 0 | 0 |
| 4352x4096x8192 | spread | rs | split | 0 | 30 | 0 | 0 |
| 4352x4096x8192 | spread | rs | split | 1 | 30 | 0 | 0 |
| 4352x4352x8192 | spread | rs | v9 | 0 | 30 | 0 | 0 |
| 4352x4352x8192 | spread | rs | v9 | 1 | 30 | 0 | 0 |
| 4352x4352x8192 | spread | rs | split | 0 | 30 | 0 | 0 |
| 4352x4352x8192 | spread | rs | split | 1 | 30 | 0 | 0 |
| 3840x4096x8192 | spread | rs | v9 | 0 | 30 | 0 | 0 |
| 3840x4096x8192 | spread | rs | v9 | 1 | 30 | 0 | 0 |
| 3840x4096x8192 | spread | rs | split | 0 | 30 | 0 | 0 |
| 3840x4096x8192 | spread | rs | split | 1 | 30 | 0 | 0 |
| 256x67328x4096 | spread | rs | v9 | 0 | 30 | 0 | 0 |
| 256x67328x4096 | spread | rs | v9 | 1 | 30 | 0 | 0 |
| 256x67328x4096 | spread | rs | split | 0 | 30 | 0 | 0 |
| 256x67328x4096 | spread | rs | split | 1 | 30 | 0 | 0 |
PASS

## STREAMK_POLICY=capped (WORKLOG entry 5), 30 rotating launches, HIP_VISIBLE_DEVICES=0

Each row is 4 configurations: tile orders v9 and split, bias 0 and 1. The rule's pick is in brackets.

| shape | configurations | wrong results | flags or counts left up |
|---|---|---|---|
| 4096x4096x8192 (S = 0, dp) | 4 | 0 | 0 |
| 4096x7168x32768 (spread) | 4 | 0 | 0 |
| 4352x4096x1024 (dp, threshold) | 4 | 0 | 0 |
| 4096x6144x2048 (dp, threshold) | 4 | 0 | 0 |
| 4096x7168x8192 (dp, S > 128) | 4 | 0 | 0 |
| 1536x11008x8192 (rs, n = 16) | 4 | 0 | 0 |
| 4352x4096x8192 (rs, n = 8) | 4 | 0 | 0 |
| 3328x5120x2048 (rs, n = 8) | 4 | 0 | 0 |
| 256x65792x8192 (rs, n = 16) | 4 | 0 | 0 |
| 256x67328x4096 (rs, n = 8) | 4 | 0 | 0 |
| 8192x8448x8192 (rs, n = 4, 4 waves) | 4 | 0 | 0 |
| 4352x4352x8192 (owner, n = 4) | 4 | 0 | 0 |
| 4096x5120x2048 (owner, n = 2) | 4 | 0 | 0 |
| 4096x5120x8192 (owner, n = 3) | 4 | 0 | 0 |
| 256x32768x8192 (owner, default split, no full wave) | 4 | 0 | 0 |
| 3840x4096x8192 (dp, no full wave) | 4 | 0 | 0 |

PASS (64 configurations)

## STREAMK_POLICY=capped (WORKLOG entry 5), 30 rotating launches, HIP_VISIBLE_DEVICES=0

Each row is 4 configurations: tile orders v9 and split, bias 0 and 1. The rule's pick is in brackets.

| shape | configurations | wrong results | flags or counts left up |
|---|---|---|---|
| 4096x4096x8192 (S = 0, dp) | 4 | 0 | 0 |
| 4096x7168x32768 (spread) | 4 | 0 | 0 |
| 4352x4096x1024 (dp, threshold) | 4 | 0 | 0 |
| 4096x6144x2048 (dp, threshold) | 4 | 0 | 0 |
| 4096x7168x8192 (dp, S > 128) | 4 | 0 | 0 |
| 1536x11008x8192 (rs, n = 16) | 4 | 0 | 0 |
| 4352x4096x8192 (rs, n = 8) | 4 | 0 | 0 |
| 3328x5120x2048 (rs, n = 8) | 4 | 0 | 0 |
| 256x65792x8192 (rs, n = 16) | 4 | 0 | 0 |
| 256x67328x4096 (rs, n = 8) | 4 | 0 | 0 |
| 8192x8448x8192 (rs, n = 4, 4 waves) | 4 | 0 | 0 |
| 4352x4352x8192 (owner, n = 4) | 4 | 0 | 0 |
| 4096x5120x2048 (owner, n = 2) | 4 | 0 | 0 |
| 4096x5120x8192 (owner, n = 3) | 4 | 0 | 0 |
| 256x32768x8192 (owner, default split, no full wave) | 4 | 0 | 0 |
| 3840x4096x8192 (dp, no full wave) | 4 | 0 | 0 |

PASS (64 configurations)
