### 256x256_4096x4096x8192

Per loop iteration (2 k-steps) and wave: 4245 cycles issued (sum of instruction latency), 2637 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 2672 | 1648 |
| other | 588 | 388 |
| ds_read | 496 | 240 |
| buffer_load | 289 | 161 |
| s_barrier | 104 | 104 |
| s_waitcnt lgkmcnt | 64 | 64 |
| s_waitcnt vmcnt+lgkmcnt | 32 | 32 |

Most stalled instructions:

- `s_waitcnt lgkmcnt(7)`: 20
- `s_barrier`: 13
- `s_barrier`: 13
- `s_barrier`: 13
- `s_barrier`: 13
- `s_barrier`: 13

### 256x128_4096x4096x8192

Per loop iteration (2 k-steps) and wave: 2538 cycles issued (sum of instruction latency), 1474 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 1441 | 853 |
| other | 398 | 232 |
| ds_read | 354 | 140 |
| s_barrier | 114 | 114 |
| buffer_load | 184 | 88 |
| s_waitcnt vmcnt+lgkmcnt | 48 | 48 |

Most stalled instructions:

- `s_waitcnt vmcnt(14) lgkmcnt(0)`: 20
- `s_barrier`: 19
- `buffer_load_dwordx4 v32, s[0:3], 0 offen lds`: 18
- `s_barrier`: 16
- `s_barrier`: 14
- `buffer_load_dwordx4 v7, s[4:7], 0 offen lds`: 14

