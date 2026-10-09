| shape | policy | fixup | partials | order | bias | launches | wrong results | atomic flags or counts left up | stale L2 flags | split tiles | tiles on 2+ XCDs | XCDs used |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3328x5120x8192 | capped | auto | l2 | v9 | 0 | 10 | error: no result after 120 s (hang) | | | | | |
| 3328x5120x8192 | capped | auto | l2 | v9 | 1 | 10 | error: no result after 120 s (hang) | | | | | |
| 1536x11008x8192 | capped | auto | l2 | v9 | 0 | 10 | error: no result after 120 s (hang) | | | | | |
| 1536x11008x8192 | capped | auto | l2 | v9 | 1 | 10 | error: no result after 120 s (hang) | | | | | |

FAIL, as required (the negative control detected the straddling placement)
