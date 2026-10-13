| spec | buffer_wbl2 | buffer_inv | atomics | write-through stores | check |
|---|---|---|---|---|---|
| v15 | 0 | 0 | 10 | 64 | ok |
| v16 | 0 | 0 | 10 | 64 | ok |
| v17 | 0 | 0 | 10 | 64 | ok |
| v18 | 0 | 0 | 10 | 64 | ok |
| v19 | 0 | 0 | 10 | 64 | ok |
| v20 policy=one_tile fixup=owner | 0 | 0 | 10 | 64 | ok |
| v20 policy=one_tile fixup=rs | 0 | 0 | 3 | 64 | ok |
| v21 policy=one_tile fixup=owner | 0 | 0 | 10 | 64 | ok |
| v21 policy=one_tile fixup=rs | 0 | 0 | 3 | 64 | ok |
| v21 policy=one_tile fixup=rs partials=mem | 2 | 2 | 3 | 64 | ok |
| v22 policy=one_tile fixup=owner | 0 | 0 | 10 | 64 | ok |
| v22 policy=one_tile fixup=rs | 0 | 0 | 3 | 64 | ok |
| v22 policy=one_tile fixup=atomic_iz | 0 | 0 | 132 | 2 | ok |
