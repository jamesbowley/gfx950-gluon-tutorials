### 256x128_4096x2048x8192

Per loop iteration (2 k-steps) and wave: 2868 cycles issued (sum of instruction latency), 1804 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 1439 | 851 |
| buffer_load | 454 | 358 |
| other | 398 | 232 |
| s_barrier | 170 | 170 |
| ds_read | 359 | 145 |
| s_waitcnt vmcnt+lgkmcnt | 48 | 48 |

Most stalled instructions:

- `buffer_load_dwordx4 v32, s[0:3], 0 offen lds`: 55
- `buffer_load_dwordx4 v6, s[4:7], 0 offen lds`: 41
- `buffer_load_dwordx4 v7, s[4:7], 0 offen lds`: 36
- `s_barrier`: 30
- `s_barrier`: 29
- `buffer_load_dwordx4 v31, s[0:3], 0 offen lds`: 28

### 256x144_4096x2304x8192

Per loop iteration (2 k-steps) and wave: 3012 cycles issued (sum of instruction latency), 1772 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 1562 | 924 |
| ds_read | 612 | 330 |
| other | 480 | 265 |
| s_barrier | 130 | 130 |
| buffer_load | 180 | 76 |
| s_waitcnt vmcnt+lgkmcnt | 48 | 48 |

Most stalled instructions:

- `s_barrier`: 24
- `s_barrier`: 23
- `s_waitcnt vmcnt(16) lgkmcnt(0)`: 20
- `s_barrier`: 19
- `s_barrier`: 13
- `s_barrier`: 13

### 256x160_4096x2560x8192

Per loop iteration (2 k-steps) and wave: 3122 cycles issued (sum of instruction latency), 1952 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 1692 | 1021 |
| other | 459 | 280 |
| buffer_load | 374 | 270 |
| ds_read | 401 | 185 |
| s_barrier | 149 | 149 |
| s_waitcnt vmcnt+lgkmcnt | 48 | 48 |

Most stalled instructions:

- `buffer_load_dwordx4 v35, s[0:3], 0 offen lds`: 47
- `buffer_load_dwordx4 v6, s[4:7], 0 offen lds`: 36
- `buffer_load_dwordx4 v7, s[4:7], 0 offen lds`: 28
- `s_barrier`: 26
- `s_barrier`: 26
- `buffer_load_dwordx4 v36, s[0:3], 0 offen lds`: 26

### 256x176_4096x2816x8192

Per loop iteration (2 k-steps) and wave: 3426 cycles issued (sum of instruction latency), 2048 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 1851 | 1103 |
| ds_read | 606 | 315 |
| other | 539 | 313 |
| s_barrier | 131 | 131 |
| buffer_load | 235 | 123 |
| s_waitcnt vmcnt+lgkmcnt | 48 | 48 |
| s_waitcnt lgkmcnt | 16 | 16 |

Most stalled instructions:

- `buffer_load_dwordx4 v6, s[0:3], 0 offen lds`: 29
- `s_barrier`: 29
- `s_waitcnt vmcnt(18) lgkmcnt(0)`: 20
- `s_barrier`: 19
- `s_barrier`: 16
- `s_barrier`: 14

### 256x192_4096x3072x8192

Per loop iteration (2 k-steps) and wave: 3523 cycles issued (sum of instruction latency), 2189 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 2107 | 1307 |
| other | 498 | 304 |
| ds_read | 444 | 217 |
| buffer_load | 263 | 150 |
| s_barrier | 132 | 132 |
| s_waitcnt lgkmcnt | 47 | 47 |
| s_waitcnt vmcnt+lgkmcnt | 32 | 32 |

Most stalled instructions:

- `buffer_load_dwordx4 v11, s[0:3], 0 offen lds`: 35
- `s_barrier`: 28
- `s_barrier`: 24
- `buffer_load_dwordx4 v35, s[0:3], 0 offen lds`: 19
- `s_waitcnt lgkmcnt(7)`: 19
- `s_barrier`: 16

### 256x208_4096x3328x8192

Per loop iteration (2 k-steps) and wave: 3837 cycles issued (sum of instruction latency), 2300 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 2241 | 1360 |
| other | 576 | 344 |
| ds_read | 616 | 312 |
| s_barrier | 137 | 137 |
| buffer_load | 203 | 83 |
| s_waitcnt vmcnt+lgkmcnt | 48 | 48 |
| s_waitcnt lgkmcnt | 16 | 16 |

Most stalled instructions:

- `s_barrier`: 22
- `s_barrier`: 22
- `s_barrier`: 22
- `buffer_load_dwordx4 v16, s[0:3], 0 offen lds`: 21
- `s_waitcnt vmcnt(18) lgkmcnt(0)`: 20
- `buffer_load_dwordx4 v15, s[0:3], 0 offen lds`: 17

### 256x224_4096x3584x8192

Per loop iteration (2 k-steps) and wave: 3831 cycles issued (sum of instruction latency), 2323 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 2381 | 1453 |
| other | 606 | 386 |
| ds_read | 447 | 207 |
| s_barrier | 111 | 111 |
| buffer_load | 207 | 87 |
| s_waitcnt lgkmcnt | 48 | 48 |
| s_waitcnt vmcnt+lgkmcnt | 32 | 32 |

Most stalled instructions:

- `s_waitcnt lgkmcnt(7)`: 20
- `s_barrier`: 17
- `s_barrier`: 14
- `s_barrier`: 14
- `s_barrier`: 13
- `s_barrier`: 13

### 256x240_4096x3840x8192

Per loop iteration (2 k-steps) and wave: 4161 cycles issued (sum of instruction latency), 2499 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 2395 | 1419 |
| other | 638 | 400 |
| ds_read | 592 | 272 |
| buffer_load | 342 | 214 |
| s_barrier | 129 | 129 |
| s_waitcnt vmcnt+lgkmcnt | 48 | 48 |
| s_waitcnt lgkmcnt | 16 | 16 |

Most stalled instructions:

- `s_barrier`: 23
- `buffer_load_dwordx4 v30, s[0:3], 0 offen lds`: 22
- `s_waitcnt vmcnt(20) lgkmcnt(0)`: 20
- `s_barrier`: 19
- `buffer_load_dwordx4 v17, s[0:3], 0 offen lds`: 17
- `s_barrier`: 17

### 256x256_4096x4096x8192

Per loop iteration (2 k-steps) and wave: 4235 cycles issued (sum of instruction latency), 2627 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 2677 | 1653 |
| other | 588 | 388 |
| ds_read | 496 | 240 |
| buffer_load | 280 | 152 |
| s_barrier | 99 | 99 |
| s_waitcnt lgkmcnt | 64 | 64 |
| s_waitcnt vmcnt+lgkmcnt | 32 | 32 |

Most stalled instructions:

- `s_waitcnt lgkmcnt(7)`: 20
- `s_barrier`: 15
- `s_barrier`: 12
- `v_mfma_f32_16x16x32_bf16 a[212:215], v[80:83], v[52:55], a[212:215]`: 12
- `v_mfma_f32_16x16x32_bf16 a[204:207], v[100:103], v[44:47], a[204:207]`: 12
- `v_mfma_f32_16x16x32_bf16 a[228:231], v[80:83], v[60:63], a[228:231]`: 12

