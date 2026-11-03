# v9_kreverse correctness (bf16)

| shape | K_ORDER | matches torch | max abs diff vs v9 | result |
|---|---|---|---|---|
| 4096x4096x8192 | 0 | True | 0 | ok |
| 4096x4096x8192 | 1 | True | 0 | ok |
| 4096x4096x8192 | 2 | True | 0 | ok |
| 8192x8192x8192 | 0 | True | 0 | ok |
| 8192x8192x8192 | 1 | True | 0 | ok |
| 8192x8192x8192 | 2 | True | 0 | ok |
| 16384x16384x8192 | 0 | True | 0 | ok |
| 16384x16384x8192 | 1 | True | 0 | ok |
| 16384x16384x8192 | 2 | True | 0 | ok |
| 8192x8192x16384 | 0 | True | 0 | ok |
| 8192x8192x16384 | 1 | True | 0 | ok |
| 8192x8192x16384 | 2 | True | 0 | ok |
| 16384x16384x16384 | 0 | True | 0 | ok |
| 16384x16384x16384 | 1 | True | 0 | ok |
| 16384x16384x16384 | 2 | True | 0 | ok |
| 6144x6144x8192 | 0 | True | 0 | ok |
| 6144x6144x8192 | 1 | True | 0 | ok |
| 6144x6144x8192 | 2 | True | 0 | ok |

The backward orders also match v9 exactly here: these bf16 inputs sum almost exactly in fp32,
so the order of the k-steps does not show. The probe below makes it show.

## K order probe (8192x8192x8192): C per tile, by dispatch wave

| K_ORDER | wave 0 | wave 1 | wave 2 | wave 3 | result |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | ok |
| 1 | 0 | 120 (backward) | 0 | 120 (backward) | ok |
| 2 | 120 (backward) | 120 (backward) | 120 (backward) | 120 (backward) | ok |

PASS
