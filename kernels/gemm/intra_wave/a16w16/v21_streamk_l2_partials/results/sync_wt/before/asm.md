| spec | buffer_wbl2 | buffer_inv | atomics | write-through stores | check |
|---|---|---|---|---|---|
| v15 | 1 | 1 | 10 | 64 | ok |
| v16 | 1 | 1 | 10 | 64 | ok |
| v17 | 1 | 1 | 10 | 64 | ok |
| v18 | 1 | 1 | 10 | 64 | ok |
| v19 | 1 | 1 | 10 | 64 | ok |
| v20 policy=one_tile fixup=owner | 1 | 1 | 10 | 64 | ok |
| v20 policy=one_tile fixup=rs | 2 | 2 | 3 | 64 | ok |
| v21 policy=one_tile fixup=owner | 1 | 1 | 10 | 64 | ok |
| v21 policy=one_tile fixup=rs | 2 | 2 | 3 | 64 | ok |
| v21 policy=one_tile fixup=rs partials=mem | 2 | 2 | 3 | 64 | ok |
| v22 policy=one_tile fixup=owner | 1 | 1 | 10 | 64 | ok |
| v22 policy=one_tile fixup=rs | 2 | 2 | 3 | 64 | ok |
| v22 policy=one_tile fixup=atomic_iz | 0 | 0 | 132 | 2 | ok |
