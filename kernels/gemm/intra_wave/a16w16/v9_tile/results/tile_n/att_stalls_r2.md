### 256x128_4096x2048x8192

Per loop iteration (2 k-steps) and wave: 2920 cycles issued (sum of instruction latency), 1856 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 1439 | 852 |
| buffer_load | 487 | 391 |
| other | 398 | 232 |
| s_barrier | 189 | 189 |
| ds_read | 359 | 145 |
| s_waitcnt vmcnt+lgkmcnt | 48 | 48 |

Most stalled instructions:

- `buffer_load_dwordx4 v6, s[4:7], 0 offen lds`: 47
- `buffer_load_dwordx4 v29, s[4:7], 0 offen lds`: 44
- `buffer_load_dwordx4 v32, s[0:3], 0 offen lds`: 39
- `buffer_load_dwordx4 v28, s[4:7], 0 offen lds`: 37
- `buffer_load_dwordx4 v7, s[4:7], 0 offen lds`: 33
- `s_barrier`: 33

### 256x144_4096x2304x8192

Per loop iteration (2 k-steps) and wave: 3039 cycles issued (sum of instruction latency), 1801 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 1562 | 924 |
| ds_read | 613 | 331 |
| other | 479 | 265 |
| s_barrier | 132 | 132 |
| buffer_load | 205 | 101 |
| s_waitcnt vmcnt+lgkmcnt | 48 | 48 |

Most stalled instructions:

- `s_barrier`: 25
- `buffer_load_dwordx4 v36, s[0:3], 0 offen lds`: 24
- `s_barrier`: 21
- `s_waitcnt vmcnt(16) lgkmcnt(0)`: 20
- `s_barrier`: 20
- `buffer_load_dwordx4 v37, s[0:3], 0 offen lds`: 15

### 256x160_4096x2560x8192

Per loop iteration (2 k-steps) and wave: 3077 cycles issued (sum of instruction latency), 1905 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 1691 | 1018 |
| other | 459 | 280 |
| buffer_load | 328 | 224 |
| ds_read | 402 | 186 |
| s_barrier | 149 | 149 |
| s_waitcnt vmcnt+lgkmcnt | 48 | 48 |

Most stalled instructions:

- `buffer_load_dwordx4 v6, s[4:7], 0 offen lds`: 31
- `buffer_load_dwordx4 v35, s[0:3], 0 offen lds`: 30
- `s_barrier`: 25
- `s_barrier`: 23
- `buffer_load_dwordx4 v7, s[4:7], 0 offen lds`: 22
- `s_barrier`: 21

### 256x176_4096x2816x8192

Per loop iteration (2 k-steps) and wave: 3345 cycles issued (sum of instruction latency), 1965 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 1851 | 1100 |
| ds_read | 606 | 314 |
| other | 538 | 312 |
| s_barrier | 112 | 112 |
| buffer_load | 175 | 62 |
| s_waitcnt vmcnt+lgkmcnt | 48 | 48 |
| s_waitcnt lgkmcnt | 16 | 16 |

Most stalled instructions:

- `s_waitcnt vmcnt(18) lgkmcnt(0)`: 20
- `s_barrier`: 18
- `s_barrier`: 15
- `s_barrier`: 14
- `s_barrier`: 13
- `s_barrier`: 13

### 256x192_4096x3072x8192

Per loop iteration (2 k-steps) and wave: 3505 cycles issued (sum of instruction latency), 2177 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 2107 | 1310 |
| other | 498 | 304 |
| ds_read | 445 | 221 |
| s_barrier | 135 | 135 |
| buffer_load | 240 | 128 |
| s_waitcnt lgkmcnt | 47 | 47 |
| s_waitcnt vmcnt+lgkmcnt | 32 | 32 |

Most stalled instructions:

- `s_barrier`: 28
- `s_barrier`: 25
- `buffer_load_dwordx4 v11, s[0:3], 0 offen lds`: 21
- `s_waitcnt lgkmcnt(7)`: 19
- `s_barrier`: 16
- `s_barrier`: 15

### 256x208_4096x3328x8192

Per loop iteration (2 k-steps) and wave: 3831 cycles issued (sum of instruction latency), 2295 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 2241 | 1360 |
| other | 576 | 344 |
| ds_read | 616 | 312 |
| s_barrier | 144 | 144 |
| buffer_load | 191 | 71 |
| s_waitcnt vmcnt+lgkmcnt | 48 | 48 |
| s_waitcnt lgkmcnt | 16 | 16 |

Most stalled instructions:

- `s_barrier`: 27
- `s_barrier`: 23
- `s_barrier`: 22
- `s_waitcnt vmcnt(18) lgkmcnt(0)`: 20
- `s_barrier`: 15
- `s_barrier`: 14

### 256x224_4096x3584x8192

Per loop iteration (2 k-steps) and wave: 3841 cycles issued (sum of instruction latency), 2333 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 2381 | 1453 |
| other | 606 | 386 |
| ds_read | 447 | 207 |
| s_barrier | 115 | 115 |
| buffer_load | 213 | 93 |
| s_waitcnt lgkmcnt | 48 | 48 |
| s_waitcnt vmcnt+lgkmcnt | 32 | 32 |

Most stalled instructions:

- `s_waitcnt lgkmcnt(7)`: 20
- `s_barrier`: 20
- `buffer_load_dwordx4 v13, s[0:3], 0 offen lds`: 15
- `s_barrier`: 15
- `s_barrier`: 15
- `s_barrier`: 14

### 256x240_4096x3840x8192

Per loop iteration (2 k-steps) and wave: 4059 cycles issued (sum of instruction latency), 2397 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 2395 | 1419 |
| other | 638 | 400 |
| ds_read | 592 | 272 |
| buffer_load | 260 | 132 |
| s_barrier | 110 | 110 |
| s_waitcnt vmcnt+lgkmcnt | 48 | 48 |
| s_waitcnt lgkmcnt | 16 | 16 |

Most stalled instructions:

- `s_waitcnt vmcnt(20) lgkmcnt(0)`: 20
- `s_barrier`: 15
- `s_barrier`: 14
- `s_barrier`: 14
- `s_barrier`: 13
- `s_barrier`: 13

### 256x256_4096x4096x8192

Per loop iteration (2 k-steps) and wave: 4243 cycles issued (sum of instruction latency), 2635 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 2677 | 1653 |
| other | 588 | 388 |
| ds_read | 496 | 240 |
| buffer_load | 286 | 158 |
| s_barrier | 100 | 100 |
| s_waitcnt lgkmcnt | 64 | 64 |
| s_waitcnt vmcnt+lgkmcnt | 32 | 32 |

Most stalled instructions:

- `s_waitcnt lgkmcnt(7)`: 20
- `s_barrier`: 16
- `s_barrier`: 12
- `s_barrier`: 12
- `v_mfma_f32_16x16x32_bf16 a[212:215], v[80:83], v[52:55], a[212:215]`: 12
- `v_mfma_f32_16x16x32_bf16 a[204:207], v[100:103], v[44:47], a[204:207]`: 12

