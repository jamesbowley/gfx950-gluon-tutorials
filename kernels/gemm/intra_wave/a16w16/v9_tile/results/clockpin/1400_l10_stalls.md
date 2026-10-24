### 256x128_4096x4096x8192

Per loop iteration (2 k-steps) and wave: 2496 cycles issued (sum of instruction latency), 1432 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 1440 | 852 |
| other | 398 | 232 |
| ds_read | 354 | 140 |
| s_barrier | 111 | 111 |
| buffer_load | 145 | 49 |
| s_waitcnt vmcnt+lgkmcnt | 48 | 48 |

Most stalled instructions:

- `s_waitcnt vmcnt(14) lgkmcnt(0)`: 20
- `s_barrier`: 19
- `s_barrier`: 17
- `s_barrier`: 13
- `s_barrier`: 13
- `s_barrier`: 13

### 256x256_4096x4096x8192

Per loop iteration (2 k-steps) and wave: 4220 cycles issued (sum of instruction latency), 2612 of them stalled.

| class | latency cycles | stall cycles |
|---|---|---|
| mfma | 2673 | 1649 |
| other | 588 | 388 |
| ds_read | 496 | 240 |
| buffer_load | 272 | 144 |
| s_barrier | 96 | 96 |
| s_waitcnt lgkmcnt | 64 | 64 |
| s_waitcnt vmcnt+lgkmcnt | 32 | 32 |

Most stalled instructions:

- `s_waitcnt lgkmcnt(7)`: 20
- `s_barrier`: 12
- `s_barrier`: 12
- `v_mfma_f32_16x16x32_bf16 a[228:231], v[80:83], v[60:63], a[228:231]`: 12
- `v_mfma_f32_16x16x32_bf16 a[220:223], v[100:103], v[52:55], a[220:223]`: 12
- `v_mfma_f32_16x16x32_bf16 a[212:215], v[80:83], v[52:55], a[212:215]`: 12

