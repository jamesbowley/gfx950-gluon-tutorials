### 256x128_4096x4096x8192

Per loop iteration (2 k-steps) and wave: 2656 cycles issued (sum of instruction latency), 1592 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 1441 | 853 |
| other | 398 | 232 |
| buffer_load | 284 | 188 |
| ds_read | 354 | 140 |
| s_barrier | 130 | 130 |
| s_waitcnt vmcnt+lgkmcnt | 48 | 48 |

Most stalled instructions:

- `buffer_load_dwordx4 v32, s[0:3], 0 offen lds`: 30
- `s_barrier`: 21
- `s_waitcnt vmcnt(14) lgkmcnt(0)`: 20
- `buffer_load_dwordx4 v6, s[4:7], 0 offen lds`: 19
- `s_barrier`: 19
- `s_barrier`: 18

### 256x256_4096x4096x8192

Per loop iteration (2 k-steps) and wave: 4282 cycles issued (sum of instruction latency), 2674 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 2673 | 1649 |
| other | 588 | 388 |
| ds_read | 496 | 240 |
| buffer_load | 317 | 189 |
| s_barrier | 113 | 113 |
| s_waitcnt lgkmcnt | 64 | 64 |
| s_waitcnt vmcnt+lgkmcnt | 32 | 32 |

Most stalled instructions:

- `s_barrier`: 25
- `s_waitcnt lgkmcnt(7)`: 20
- `buffer_load_dwordx4 v5, s[0:3], 0 offen lds`: 18
- `buffer_load_dwordx4 v4, s[0:3], 0 offen lds`: 17
- `buffer_load_dwordx4 v3, s[0:3], 0 offen lds`: 15
- `s_barrier`: 15

