Stress check after removing the no-op `buffer_inv sc0` from the `l2` tile barrier (GPU 1,
published stack: `check_streamk.py --shapes 3072x5632x8192 4352x4096x8192 3328x5120x8192
--partials l2 --policies capped one_tile --launches 50`).

| shape | policy | fixup | partials | order | bias | launches | wrong results | atomic flags or counts left up | stale L2 flags | split tiles | tiles on 2+ XCDs | XCDs used |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3072x5632x8192 | capped | auto | l2 | v9 | 0 | 50 | 0 | 0 | 0 | 8 | 0 | 8 |
| 3072x5632x8192 | capped | auto | l2 | v9 | 1 | 50 | 0 | 0 | 0 | 8 | 0 | 8 |
| 3072x5632x8192 | one_tile | rs | l2 | v9 | 0 | 50 | 0 | 0 | 0 | 8 | 0 | 8 |
| 3072x5632x8192 | one_tile | rs | l2 | v9 | 1 | 50 | 0 | 0 | 0 | 8 | 0 | 8 |
| 4352x4096x8192 | capped | auto | l2 | v9 | 0 | 50 | 0 | 0 | 0 | 16 | 0 | 8 |
| 4352x4096x8192 | capped | auto | l2 | v9 | 1 | 50 | 0 | 0 | 0 | 16 | 0 | 8 |
| 4352x4096x8192 | one_tile | rs | l2 | v9 | 0 | 50 | 0 | 0 | 0 | 16 | 0 | 8 |
| 4352x4096x8192 | one_tile | rs | l2 | v9 | 1 | 50 | 0 | 0 | 0 | 16 | 0 | 8 |
| 3328x5120x8192 | capped | auto | l2 | v9 | 0 | 50 | 0 | 0 | 0 | 4 | 0 | 4 |
| 3328x5120x8192 | capped | auto | l2 | v9 | 1 | 50 | 0 | 0 | 0 | 4 | 0 | 4 |
| 3328x5120x8192 | one_tile | rs | l2 | v9 | 0 | 50 | 0 | 0 | 0 | 4 | 0 | 4 |
| 3328x5120x8192 | one_tile | rs | l2 | v9 | 1 | 50 | 0 | 0 | 0 | 4 | 0 | 4 |

PASS
