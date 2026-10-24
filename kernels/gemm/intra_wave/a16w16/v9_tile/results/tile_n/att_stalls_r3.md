### 256x128_4096x2048x8192

Per loop iteration (2 k-steps) and wave: 2697 cycles issued (sum of instruction latency), 1633 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 1438 | 850 |
| other | 398 | 232 |
| buffer_load | 309 | 213 |
| ds_read | 359 | 145 |
| s_barrier | 145 | 145 |
| s_waitcnt vmcnt+lgkmcnt | 48 | 48 |

Most stalled instructions:

- `buffer_load_dwordx4 v32, s[0:3], 0 offen lds`: 41
- `buffer_load_dwordx4 v7, s[4:7], 0 offen lds`: 34
- `buffer_load_dwordx4 v6, s[4:7], 0 offen lds`: 31
- `s_barrier`: 28
- `s_barrier`: 22
- `s_barrier`: 22

### 256x144_4096x2304x8192

Per loop iteration (2 k-steps) and wave: 3087 cycles issued (sum of instruction latency), 1842 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 1563 | 923 |
| ds_read | 613 | 327 |
| other | 480 | 265 |
| buffer_load | 249 | 145 |
| s_barrier | 134 | 134 |
| s_waitcnt vmcnt+lgkmcnt | 48 | 48 |

Most stalled instructions:

- `buffer_load_dwordx4 v19, s[4:7], 0 offen lds`: 36
- `s_barrier`: 24
- `buffer_load_dwordx4 v18, s[4:7], 0 offen lds`: 23
- `s_barrier`: 22
- `s_waitcnt vmcnt(16) lgkmcnt(0)`: 20
- `s_barrier`: 19

### 256x160_4096x2560x8192

Per loop iteration (2 k-steps) and wave: 3119 cycles issued (sum of instruction latency), 1948 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 1691 | 1020 |
| other | 459 | 280 |
| buffer_load | 364 | 260 |
| ds_read | 401 | 185 |
| s_barrier | 156 | 156 |
| s_waitcnt vmcnt+lgkmcnt | 48 | 48 |

Most stalled instructions:

- `buffer_load_dwordx4 v35, s[0:3], 0 offen lds`: 36
- `s_barrier`: 28
- `buffer_load_dwordx4 v6, s[4:7], 0 offen lds`: 25
- `s_barrier`: 23
- `buffer_load_dwordx4 v36, s[0:3], 0 offen lds`: 23
- `s_barrier`: 23

### 256x176_4096x2816x8192

Per loop iteration (2 k-steps) and wave: 3357 cycles issued (sum of instruction latency), 1976 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 1851 | 1100 |
| ds_read | 607 | 315 |
| other | 538 | 312 |
| s_barrier | 116 | 116 |
| buffer_load | 182 | 69 |
| s_waitcnt vmcnt+lgkmcnt | 48 | 48 |
| s_waitcnt lgkmcnt | 16 | 16 |

Most stalled instructions:

- `s_waitcnt vmcnt(18) lgkmcnt(0)`: 20
- `s_barrier`: 19
- `s_barrier`: 15
- `s_barrier`: 15
- `s_barrier`: 14
- `s_barrier`: 13

### 256x192_4096x3072x8192

Per loop iteration (2 k-steps) and wave: 3456 cycles issued (sum of instruction latency), 2128 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 2106 | 1309 |
| other | 498 | 304 |
| ds_read | 446 | 222 |
| s_barrier | 119 | 119 |
| buffer_load | 208 | 96 |
| s_waitcnt lgkmcnt | 47 | 47 |
| s_waitcnt vmcnt+lgkmcnt | 32 | 32 |

Most stalled instructions:

- `s_barrier`: 24
- `buffer_load_dwordx4 v11, s[0:3], 0 offen lds`: 19
- `s_waitcnt lgkmcnt(7)`: 19
- `s_barrier`: 16
- `s_barrier`: 16
- `s_barrier`: 15

### 256x208_4096x3328x8192

Per loop iteration (2 k-steps) and wave: 3818 cycles issued (sum of instruction latency), 2281 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 2241 | 1360 |
| other | 576 | 344 |
| ds_read | 616 | 312 |
| s_barrier | 135 | 135 |
| buffer_load | 186 | 66 |
| s_waitcnt vmcnt+lgkmcnt | 48 | 48 |
| s_waitcnt lgkmcnt | 16 | 16 |

Most stalled instructions:

- `s_barrier`: 22
- `s_barrier`: 22
- `s_waitcnt vmcnt(18) lgkmcnt(0)`: 20
- `s_barrier`: 19
- `s_barrier`: 16
- `s_barrier`: 14

### 256x224_4096x3584x8192

Per loop iteration (2 k-steps) and wave: 3860 cycles issued (sum of instruction latency), 2352 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 2381 | 1453 |
| other | 606 | 386 |
| ds_read | 447 | 207 |
| s_barrier | 125 | 125 |
| buffer_load | 222 | 102 |
| s_waitcnt lgkmcnt | 48 | 48 |
| s_waitcnt vmcnt+lgkmcnt | 32 | 32 |

Most stalled instructions:

- `s_barrier`: 21
- `s_waitcnt lgkmcnt(7)`: 20
- `s_barrier`: 19
- `s_barrier`: 18
- `s_barrier`: 14
- `s_barrier`: 14

### 256x240_4096x3840x8192

Per loop iteration (2 k-steps) and wave: 4047 cycles issued (sum of instruction latency), 2385 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 2395 | 1419 |
| other | 638 | 400 |
| ds_read | 592 | 272 |
| buffer_load | 255 | 127 |
| s_barrier | 104 | 104 |
| s_waitcnt vmcnt+lgkmcnt | 48 | 48 |
| s_waitcnt lgkmcnt | 16 | 16 |

Most stalled instructions:

- `s_waitcnt vmcnt(20) lgkmcnt(0)`: 20
- `s_barrier`: 14
- `s_barrier`: 13
- `s_barrier`: 13
- `s_barrier`: 13
- `s_barrier`: 13

### 256x256_4096x4096x8192

Per loop iteration (2 k-steps) and wave: 4229 cycles issued (sum of instruction latency), 2621 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 2677 | 1653 |
| other | 588 | 388 |
| ds_read | 496 | 240 |
| buffer_load | 275 | 147 |
| s_barrier | 97 | 97 |
| s_waitcnt lgkmcnt | 64 | 64 |
| s_waitcnt vmcnt+lgkmcnt | 32 | 32 |

Most stalled instructions:

- `s_waitcnt lgkmcnt(7)`: 20
- `s_barrier`: 13
- `s_barrier`: 12
- `v_mfma_f32_16x16x32_bf16 a[204:207], v[100:103], v[44:47], a[204:207]`: 12
- `v_mfma_f32_16x16x32_bf16 a[228:231], v[80:83], v[60:63], a[228:231]`: 12
- `v_mfma_f32_16x16x32_bf16 a[220:223], v[100:103], v[52:55], a[220:223]`: 12

