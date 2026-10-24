### Efficiency, TFLOPS and clock

| tile | shape | ATT loop eff (median [range]) | ATT whole wave | loop x share | ATT GHz | ATT TFLOPS | ATT chip-wide | window at ATT clock | isolated TFLOPS | back-to-back TFLOPS | hwmon GHz | W | back-to-back chip-wide | counters MFMA busy | window at hwmon clock | L2 hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256x256 | 4096x4096x8192 | 95.6% [95.5%, 95.8%] | 90.7% [90.5%, 90.8%] | 89.5% | 1.44 | 1237 | 82.8% | 100% | 1572 | 1579 | 1.80 | 1400 | 83.5% | 80.0% | 100% | 80.3% |
| 256x240 | 4096x3840x8192 | 92.2% [90.0%, 92.5%] | 87.4% [85.3%, 87.5%] | 86.2% | 1.42 | 1169 | 77.9% | 100% | 1512 | 1500 | 1.74 | 1400 | 82.3% | 76.4% | 100% | 80.6% |
| 256x224 | 4096x3584x8192 | 91.8% [91.4%, 92.1%] | 87.3% [86.8%, 87.5%] | 86.1% | 1.48 | 1238 | 79.8% | 100% | 1523 | 1515 | 1.75 | 1400 | 82.4% | 76.5% | 99% | 80.8% |
| 256x208 | 4096x3328x8192 | 85.1% [85.0%, 85.4%] | 81.0% [80.8%, 81.1%] | 79.9% | 1.56 | 1206 | 74.1% | 100% | 1422 | 1430 | 1.83 | 1400 | 74.7% | 71.1% | 91% | 81.2% |
| 256x192 | 4096x3072x8192 | 86.7% [86.2%, 87.9%] | 82.3% [81.8%, 83.4%] | 81.2% | 1.52 | 1229 | 75.8% | 100% | 1447 | 1442 | 1.82 | 1400 | 75.3% | 71.2% | 87% | 81.3% |
| 256x176 | 4096x2816x8192 | 82.3% [80.6%, 82.6%] | 78.2% [76.8%, 78.5%] | 77.1% | 1.46 | 1094 | 71.3% | 100% | 1372 | 1376 | 1.84 | 1399 | 71.4% | 67.3% | 83% | 81.8% |
| 256x160 | 4096x2560x8192 | 81.6% [81.5%, 82.8%] | 77.7% [77.2%, 78.6%] | 76.6% | 1.75 | 1298 | 70.6% | 82% | 1374 | 1382 | 1.82 | 1397 | 72.4% | 65.9% | 79% | 82.0% |
| 256x144 | 4096x2304x8192 | 74.7% [73.6%, 75.4%] | 71.2% [70.0%, 71.7%] | 70.2% | 1.68 | 1127 | 64.0% | 80% | 1262 | 1278 | 1.86 | 1400 | 65.6% | 60.6% | 72% | 82.4% |
| 256x128 | 4096x2048x8192 | 71.7% [70.4%, 76.3%] | 68.6% [67.4%, 72.8%] | 67.7% | 1.75 | 1149 | 62.4% | 71% | 1224 | 1243 | 1.90 | 1398 | 62.4% | 55.3% | 65% | 82.7% |

### The compiled loop (one iteration = two k-steps, per wave)

| tile | loop insts | MFMAs | non-MFMA | non-MFMA per MFMA | buffer loads | ds_reads | s_waitcnt | s_nop | longest non-MFMA run | VGPRs | AGPRs | LDS KB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256x256 | 527 | 256 | 271 | 1.06 | 32 | 64 | 20 | 97 | 5 | 456 | 256 | 131.9 |
| 256x240 | 522 | 240 | 282 | 1.18 | 32 | 80 | 12 | 92 | 6 | 468 | 256 | 127.6 |
| 256x224 | 480 | 224 | 256 | 1.14 | 30 | 60 | 16 | 90 | 7 | 424 | 256 | 123.6 |
| 256x208 | 470 | 208 | 262 | 1.26 | 30 | 76 | 12 | 82 | 12 | 456 | 256 | 119.3 |
| 256x192 | 422 | 192 | 230 | 1.20 | 28 | 56 | 16 | 76 | 15 | 412 | 256 | 115.3 |
| 256x176 | 426 | 176 | 250 | 1.42 | 28 | 72 | 12 | 78 | 22 | 400 | 208 | 111.1 |
| 256x160 | 366 | 160 | 206 | 1.29 | 26 | 52 | 8 | 70 | 15 | 428 | 200 | 107.1 |
| 256x144 | 368 | 144 | 224 | 1.56 | 26 | 68 | 8 | 66 | 19 | 432 | 180 | 102.9 |
| 256x128 | 311 | 128 | 183 | 1.43 | 24 | 48 | 8 | 58 | 13 | 348 | 160 | 98.9 |

### MFMA idle per loop iteration and wave, by what was exposed (att_gaps.py)

| tile | repeat | cycles / iter | MFMA idle / iter | issue | dependency hole | lds: ds_read stall | lds: lgkmcnt wait | barrier | other stall | memory: vmcnt wait | memory: buffer_load issue stall | mfma stall, pipe empty | unattributed | regions that fit but idle (idle cycles) | regions that do not fit | max barrier skew |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256x256 | 1 | 4283 | 187 | 69 | 48 | 8 | 20 | 19 | 16 | 0 | 7 | 1 | 0 | 8/8 (156) | 0 | 15 |
| 256x256 | 2 | 4291 | 195 | 70 | 48 | 8 | 20 | 19 | 17 | 0 | 12 | 1 | 0 | 8/8 (163) | 0 | 18 |
| 256x256 | 3 | 4277 | 181 | 69 | 48 | 8 | 20 | 17 | 16 | 0 | 2 | 1 | 0 | 8/8 (152) | 0 | 14 |
| 256x240 | 1 | 4267 | 427 | 133 | 80 | 30 | 0 | 47 | 27 | 20 | 87 | 3 | 0 | 8/8 (367) | 0 | 32 |
| 256x240 | 2 | 4165 | 325 | 128 | 80 | 29 | 0 | 30 | 25 | 20 | 10 | 3 | 0 | 8/8 (283) | 0 | 15 |
| 256x240 | 3 | 4153 | 313 | 125 | 80 | 29 | 0 | 25 | 23 | 20 | 8 | 3 | 0 | 8/8 (276) | 0 | 16 |
| 256x224 | 1 | 3891 | 307 | 119 | 55 | 28 | 20 | 36 | 33 | 0 | 16 | 1 | 0 | 8/10 (271) | 0 | 20 |
| 256x224 | 2 | 3901 | 317 | 119 | 55 | 28 | 20 | 40 | 34 | 0 | 21 | 1 | 0 | 8/10 (277) | 0 | 26 |
| 256x224 | 3 | 3920 | 336 | 119 | 55 | 28 | 20 | 50 | 34 | 0 | 29 | 1 | -0 | 8/10 (286) | 0 | 26 |
| 256x208 | 1 | 3917 | 589 | 206 | 80 | 150 | 0 | 67 | 27 | 20 | 38 | 1 | 0 | 8/10 (510) | 0 | 31 |
| 256x208 | 2 | 3911 | 583 | 205 | 80 | 150 | 0 | 74 | 26 | 20 | 27 | 1 | 0 | 8/10 (497) | 0 | 37 |
| 256x208 | 3 | 3898 | 570 | 205 | 80 | 150 | 0 | 65 | 27 | 20 | 22 | 1 | -0 | 8/10 (493) | 0 | 31 |
| 256x192 | 1 | 3562 | 490 | 171 | 43 | 95 | 19 | 60 | 25 | 0 | 72 | 6 | 0 | 8/10 (431) | 0 | 43 |
| 256x192 | 2 | 3544 | 472 | 169 | 43 | 96 | 19 | 63 | 25 | 0 | 51 | 6 | 0 | 8/10 (409) | 0 | 39 |
| 256x192 | 3 | 3495 | 423 | 168 | 43 | 96 | 19 | 46 | 24 | 0 | 20 | 6 | 0 | 8/10 (376) | 0 | 35 |
| 256x176 | 1 | 3492 | 676 | 262 | 80 | 139 | 0 | 70 | 26 | 20 | 76 | 3 | 0 | 8/8 (606) | 0 | 42 |
| 256x176 | 2 | 3411 | 595 | 259 | 80 | 138 | 0 | 51 | 25 | 20 | 19 | 3 | 0 | 8/8 (544) | 0 | 23 |
| 256x176 | 3 | 3423 | 607 | 260 | 80 | 139 | 0 | 55 | 26 | 20 | 25 | 3 | 0 | 8/8 (552) | 0 | 26 |
| 256x160 | 1 | 3139 | 579 | 171 | 25 | 61 | 0 | 84 | 24 | 20 | 193 | 3 | -1 | 8/8 (497) | 0 | 39 |
| 256x160 | 2 | 3094 | 534 | 176 | 25 | 63 | 0 | 81 | 26 | 20 | 142 | 2 | -2 | 8/8 (454) | 0 | 32 |
| 256x160 | 3 | 3136 | 576 | 171 | 25 | 61 | 0 | 90 | 24 | 20 | 183 | 3 | -1 | 8/8 (487) | 0 | 39 |
| 256x144 | 1 | 3057 | 753 | 298 | 63 | 203 | 0 | 74 | 29 | 20 | 64 | 1 | -0 | 8/8 (679) | 0 | 33 |
| 256x144 | 2 | 3084 | 780 | 297 | 63 | 204 | 0 | 76 | 29 | 20 | 90 | 1 | -0 | 8/8 (704) | 0 | 35 |
| 256x144 | 3 | 3132 | 828 | 300 | 63 | 202 | 0 | 78 | 29 | 20 | 135 | 1 | -0 | 8/8 (750) | 0 | 35 |
| 256x128 | 1 | 2856 | 808 | 218 | 6 | 80 | 0 | 114 | 28 | 20 | 341 | 6 | -4 | 8/8 (698) | 0 | 45 |
| 256x128 | 2 | 2908 | 860 | 219 | 6 | 80 | 0 | 133 | 28 | 20 | 372 | 7 | -4 | 8/8 (731) | 0 | 46 |
| 256x128 | 3 | 2684 | 636 | 212 | 6 | 80 | 0 | 89 | 28 | 20 | 201 | 5 | -4 | 8/8 (551) | 0 | 40 |

### Traces to investigate

Thresholds: window model gap 5%, repeat spread 3%, whole wave vs loop x share 2%, chip-wide vs whole wave 5%, one instruction 3% of MFMA idle, vmcnt 5%, unattributed 20%, monotonic 1%.

#### 256x256 (loop 95.6%, 95.5%, 95.8%; ATT clock 1.44, 1.53, 1.30 GHz)

- Repeat 1: `v9_tile/tmp/tile_n/att_r1/sweep_att/256x256_4096x4096x8192/ui_output_agent_62853_dispatch_58` (wave files se0_sm0_sl0_wv0.json, se0_sm1_sl0_wv0.json, se0_sm2_sl0_wv0.json, se0_sm3_sl0_wv0.json, se2_sm0_sl0_wv0.json, se2_sm1_sl0_wv0.json, se2_sm2_sl0_wv0.json, se2_sm3_sl0_wv0.json, se3_sm0_sl0_wv0.json, se3_sm1_sl0_wv0.json, se3_sm2_sl0_wv0.json, se3_sm3_sl0_wv0.json)
  - chip-wide 82.2% < whole wave 90.8%
  - 7 instruction(s) over 3% of MFMA idle each (largest 10.6%)
- Repeat 2: `v9_tile/tmp/tile_n/att_r2/sweep_att/256x256_4096x4096x8192/ui_output_agent_17729_dispatch_58`
  - chip-wide 82.8% < whole wave 90.5%
  - 8 instruction(s) over 3% of MFMA idle each (largest 10.1%)
- Repeat 3: `v9_tile/tmp/tile_n/att_r3/sweep_att/256x256_4096x4096x8192/ui_output_agent_33290_dispatch_58`
  - chip-wide 83.5% < whole wave 90.7%
  - 7 instruction(s) over 3% of MFMA idle each (largest 10.9%)
- Repeat 1 (median): MFMA idle 187 of 4283 cycles per iteration (issue 37%, dependency hole 26%, lds: lgkmcnt wait 11%, barrier 10%); 8/8 MFMA regions fit their shadow yet idle (156 cycles); max barrier skew 15 cycles.
- Hot spots, repeat 1 (exposed cycles per iteration and wave):
  - code index 634: `s_waitcnt lgkmcnt(7)` (s_waitcnt lgkmcnt): 0.0 hole + 19.7 stall + 0.0 issue, 10.6% of MFMA idle
  - code index 774: `ds_read_b128 v[64:67], v39` after `v_add_u32_e32 v39, 0x149c0, v1` (ds_read): 8.0 hole + 2.0 stall + 4.0 issue, 7.5% of MFMA idle
  - code index 967: `ds_read_b128 v[144:147], v39` after `v_add_u32_e32 v39, 0x1cda0, v1` (ds_read): 8.0 hole + 2.0 stall + 4.0 issue, 7.5% of MFMA idle
  - code index 1030: `ds_read_b128 v[96:99], v39` after `v_add_u32_e32 v39, 0x107c0, v1` (ds_read): 8.0 hole + 2.0 stall + 4.0 issue, 7.5% of MFMA idle
  - code index 707: `ds_read_b128 v[144:147], v39` after `v_add_u32_e32 v39, 0x18ba0, v1` (ds_read): 8.0 hole + 2.0 stall + 4.0 issue, 7.5% of MFMA idle
  - code index 635: `v_mfma_f32_16x16x32_bf16 a[252:255], v[96:99], v[88:91], a[252:255]` after `s_waitcnt lgkmcnt(7)` (mfma): 11.8 hole + 0.0 stall + 0.0 issue, 6.3% of MFMA idle
  - wait at code index 634 `s_waitcnt lgkmcnt(7)` (lgkmcnt): stalls 20 cycles 0.98x per iteration, 490 cycles (29 MFMAs) after the load it waits on

#### 256x240 (loop 90.0%, 92.2%, 92.5%; ATT clock 1.62, 1.41, 1.42 GHz)

- Repeat 1: `v9_tile/tmp/tile_n/att_r1/sweep_att/256x240_4096x3840x8192/ui_output_agent_22804_dispatch_60`
  - loop 90.0% vs window ceiling 100% at 1.62 GHz (under by 10.0 points)
  - chip-wide 76.6% < whole wave 85.3%
  - 6 instruction(s) over 3% of MFMA idle each (largest 5.2%)
- Repeat 2: `v9_tile/tmp/tile_n/att_r2/sweep_att/256x240_4096x3840x8192/ui_output_agent_38965_dispatch_60` (wave files se0_sm0_sl0_wv0.json, se0_sm1_sl0_wv0.json, se0_sm2_sl0_wv0.json, se0_sm3_sl0_wv0.json, se2_sm0_sl0_wv0.json, se2_sm1_sl0_wv0.json, se2_sm2_sl0_wv0.json, se2_sm3_sl0_wv0.json, se3_sm0_sl0_wv0.json, se3_sm1_sl0_wv0.json, se3_sm2_sl0_wv0.json, se3_sm3_sl0_wv0.json)
  - loop 92.2% vs window ceiling 100% at 1.41 GHz (under by 7.8 points)
  - chip-wide 79.1% < whole wave 87.4%
  - 6 instruction(s) over 3% of MFMA idle each (largest 6.1%)
  - vmcnt waits 6% of MFMA idle
- Repeat 3: `v9_tile/tmp/tile_n/att_r3/sweep_att/256x240_4096x3840x8192/ui_output_agent_48617_dispatch_60`
  - loop 92.5% vs window ceiling 100% at 1.42 GHz (under by 7.6 points)
  - chip-wide 77.9% < whole wave 87.5%
  - 6 instruction(s) over 3% of MFMA idle each (largest 6.3%)
  - vmcnt waits 6% of MFMA idle
- Repeat 2 (median): MFMA idle 325 of 4165 cycles per iteration (issue 39%, dependency hole 25%, barrier 9%, lds: ds_read stall 9%); 8/8 MFMA regions fit their shadow yet idle (283 cycles); max barrier skew 15 cycles.
- Hot spots, repeat 2 (exposed cycles per iteration and wave):
  - code index 658: `s_waitcnt vmcnt(20) lgkmcnt(0)` (s_waitcnt vmcnt+lgkmcnt): 0.0 hole + 19.7 stall + 0.0 issue, 6.1% of MFMA idle
  - code index 798: `ds_read_b128 v[112:115], v0` after `v_add_u32_e32 v0, 0x128c0, v14` (ds_read): 8.0 hole + 2.0 stall + 4.0 issue, 4.3% of MFMA idle
  - code index 990: `ds_read_b128 v[192:195], v0` after `v_add_u32_e32 v0, 0x16aa0, v14` (ds_read): 8.0 hole + 2.0 stall + 4.0 issue, 4.3% of MFMA idle
  - code index 1052: `ds_read_b128 v[120:123], v0` after `v_add_u32_e32 v0, 0x107c0, v14` (ds_read): 8.0 hole + 2.0 stall + 4.0 issue, 4.3% of MFMA idle
  - code index 732: `ds_read_b128 v[192:195], v0` after `v_add_u32_e32 v0, 0x149a0, v14` (ds_read): 8.0 hole + 2.0 stall + 4.0 issue, 4.3% of MFMA idle
  - code index 659: `v_mfma_f32_16x16x32_bf16 a[236:239], v[120:123], v[92:95], a[236:239]` after `s_waitcnt vmcnt(20) lgkmcnt(0)` (mfma): 11.8 hole + 0.0 stall + 0.0 issue, 3.6% of MFMA idle
  - wait at code index 658 `s_waitcnt vmcnt(20) lgkmcnt(0)` (vmcnt): stalls 20 cycles 0.98x per iteration, 2653 cycles (154 MFMAs) after the load it waits on
  - wait at code index 658 `s_waitcnt vmcnt(20) lgkmcnt(0)` (lgkmcnt): stalls 20 cycles 0.98x per iteration, 335 cycles (19 MFMAs) after the load it waits on

#### 256x224 (loop 92.1%, 91.8%, 91.4%; ATT clock 1.34, 1.48, 1.49 GHz)

- Repeat 1: `v9_tile/tmp/tile_n/att_r1/sweep_att/256x224_4096x3584x8192/ui_output_agent_37173_dispatch_60`
  - loop 92.1% vs window ceiling 100% at 1.34 GHz (under by 7.9 points)
  - chip-wide 81.3% < whole wave 87.5%
  - 7 instruction(s) over 3% of MFMA idle each (largest 6.4%)
- Repeat 2: `v9_tile/tmp/tile_n/att_r2/sweep_att/256x224_4096x3584x8192/ui_output_agent_11490_dispatch_60` (wave files se0_sm0_sl0_wv0.json, se0_sm1_sl0_wv0.json, se0_sm2_sl0_wv0.json, se0_sm3_sl0_wv0.json, se2_sm0_sl0_wv0.json, se2_sm1_sl0_wv0.json, se2_sm2_sl0_wv0.json, se2_sm3_sl0_wv0.json, se3_sm0_sl0_wv0.json, se3_sm1_sl0_wv0.json, se3_sm2_sl0_wv0.json, se3_sm3_sl0_wv0.json)
  - loop 91.8% vs window ceiling 100% at 1.48 GHz (under by 8.2 points)
  - chip-wide 79.8% < whole wave 87.3%
  - 9 instruction(s) over 3% of MFMA idle each (largest 6.2%)
- Repeat 3: `v9_tile/tmp/tile_n/att_r3/sweep_att/256x224_4096x3584x8192/ui_output_agent_29007_dispatch_60`
  - loop 91.4% vs window ceiling 100% at 1.49 GHz (under by 8.6 points)
  - chip-wide 79.2% < whole wave 86.8%
  - 6 instruction(s) over 3% of MFMA idle each (largest 5.9%)
- Repeat 2 (median): MFMA idle 317 of 3901 cycles per iteration (issue 37%, dependency hole 17%, barrier 13%, other stall 11%); 8/10 MFMA regions fit their shadow yet idle (277 cycles); max barrier skew 26 cycles.
- Hot spots, repeat 2 (exposed cycles per iteration and wave):
  - code index 604: `s_waitcnt lgkmcnt(7)` (s_waitcnt lgkmcnt): 0.0 hole + 19.7 stall + 0.0 issue, 6.2% of MFMA idle
  - code index 679: `ds_read_b128 v[146:149], v41` after `v_add_u32_e32 v41, 0x149a0, v12` (ds_read): 8.0 hole + 2.0 stall + 4.0 issue, 4.4% of MFMA idle
  - code index 740: `ds_read_b128 v[94:97], v41` after `v_add_u32_e32 v41, 0x128c0, v12` (ds_read): 8.0 hole + 2.0 stall + 4.0 issue, 4.4% of MFMA idle
  - code index 916: `ds_read_b128 v[150:153], v41` after `v_add_u32_e32 v41, 0x16aa0, v12` (ds_read): 8.0 hole + 2.0 stall + 4.0 issue, 4.4% of MFMA idle
  - code index 973: `ds_read_b128 v[98:101], v41` after `v_add_u32_e32 v41, 0x107c0, v12` (ds_read): 8.0 hole + 2.0 stall + 4.0 issue, 4.4% of MFMA idle
  - code index 676: `s_barrier` (s_barrier): 0.0 hole + 11.7 stall + 0.0 issue, 3.7% of MFMA idle
  - wait at code index 604 `s_waitcnt lgkmcnt(7)` (lgkmcnt): stalls 20 cycles 0.98x per iteration, 396 cycles (23 MFMAs) after the load it waits on

#### 256x208 (loop 85.0%, 85.1%, 85.4%; ATT clock 1.61, 1.46, 1.56 GHz)

- Repeat 1: `v9_tile/tmp/tile_n/att_r1/sweep_att/256x208_4096x3328x8192/ui_output_agent_10815_dispatch_62`
  - non-monotonic: 85.1% below the narrower 256x192's 86.7%
  - loop 85.0% vs window ceiling 100% at 1.61 GHz (under by 15.0 points)
  - chip-wide 74.1% < whole wave 80.8%
  - 1 instruction(s) over 3% of MFMA idle each (largest 3.4%)
- Repeat 2: `v9_tile/tmp/tile_n/att_r2/sweep_att/256x208_4096x3328x8192/ui_output_agent_4147_dispatch_62` (wave files se0_sm0_sl0_wv0.json, se0_sm1_sl0_wv0.json, se0_sm2_sl0_wv0.json, se0_sm3_sl0_wv0.json, se2_sm0_sl0_wv0.json, se2_sm1_sl0_wv0.json, se2_sm2_sl0_wv0.json, se2_sm3_sl0_wv0.json, se3_sm0_sl0_wv0.json, se3_sm1_sl0_wv0.json, se3_sm2_sl0_wv0.json, se3_sm3_sl0_wv0.json)
  - loop 85.1% vs window ceiling 100% at 1.46 GHz (under by 14.9 points)
  - 2 instruction(s) over 3% of MFMA idle each (largest 3.4%)
- Repeat 3: `v9_tile/tmp/tile_n/att_r3/sweep_att/256x208_4096x3328x8192/ui_output_agent_62483_dispatch_62`
  - loop 85.4% vs window ceiling 100% at 1.56 GHz (under by 14.6 points)
  - chip-wide 73.9% < whole wave 81.1%
  - 1 instruction(s) over 3% of MFMA idle each (largest 3.5%)
- Repeat 2 (median): MFMA idle 583 of 3911 cycles per iteration (issue 35%, lds: ds_read stall 26%, dependency hole 14%, barrier 13%); 8/10 MFMA regions fit their shadow yet idle (497 cycles); max barrier skew 37 cycles.
- Hot spots, repeat 2 (exposed cycles per iteration and wave):
  - code index 616: `s_waitcnt vmcnt(18) lgkmcnt(0)` (s_waitcnt vmcnt+lgkmcnt): 0.0 hole + 19.7 stall + 0.0 issue, 3.4% of MFMA idle
  - code index 1030: `s_barrier` (s_barrier): 0.0 hole + 19.0 stall + 0.0 issue, 3.3% of MFMA idle
  - code index 809: `ds_read_b128 v[44:47], v1 offset:17664` (ds_read): 0.0 hole + 12.0 stall + 4.0 issue, 2.7% of MFMA idle
  - code index 1037: `ds_read_b128 v[60:63], v1 offset:768` (ds_read): 0.0 hole + 12.0 stall + 4.0 issue, 2.7% of MFMA idle
  - code index 689: `s_barrier` (s_barrier): 0.0 hole + 15.5 stall + 0.0 issue, 2.7% of MFMA idle
  - code index 808: `ds_read_b128 v[64:67], v1 offset:17472` (ds_read): 0.0 hole + 11.0 stall + 4.0 issue, 2.6% of MFMA idle
  - wait at code index 616 `s_waitcnt vmcnt(18) lgkmcnt(0)` (vmcnt): stalls 20 cycles 0.98x per iteration, 2437 cycles (130 MFMAs) after the load it waits on
  - wait at code index 616 `s_waitcnt vmcnt(18) lgkmcnt(0)` (lgkmcnt): stalls 20 cycles 0.98x per iteration, 268 cycles (15 MFMAs) after the load it waits on

#### 256x192 (loop 86.2%, 86.7%, 87.9%; ATT clock 1.72, 1.52, 1.45 GHz)

- Repeat 1: `v9_tile/tmp/tile_n/att_r1/sweep_att/256x192_4096x3072x8192/ui_output_agent_64795_dispatch_64`
  - loop 86.2% vs window ceiling 93% at 1.72 GHz (under by 6.5 points)
  - chip-wide 74.7% < whole wave 81.8%
  - 4 instruction(s) over 3% of MFMA idle each (largest 6.2%)
- Repeat 2: `v9_tile/tmp/tile_n/att_r2/sweep_att/256x192_4096x3072x8192/ui_output_agent_36703_dispatch_64` (wave files se0_sm0_sl0_wv0.json, se0_sm1_sl0_wv0.json, se0_sm2_sl0_wv0.json, se0_sm3_sl0_wv0.json, se2_sm0_sl0_wv0.json, se2_sm1_sl0_wv0.json, se2_sm2_sl0_wv0.json, se2_sm3_sl0_wv0.json, se3_sm0_sl0_wv0.json, se3_sm1_sl0_wv0.json, se3_sm2_sl0_wv0.json, se3_sm3_sl0_wv0.json)
  - loop 86.7% vs window ceiling 100% at 1.52 GHz (under by 13.3 points)
  - chip-wide 77.2% < whole wave 82.3%
  - 4 instruction(s) over 3% of MFMA idle each (largest 4.3%)
- Repeat 3: `v9_tile/tmp/tile_n/att_r3/sweep_att/256x192_4096x3072x8192/ui_output_agent_20780_dispatch_64`
  - loop 87.9% vs window ceiling 100% at 1.45 GHz (under by 12.1 points)
  - chip-wide 75.8% < whole wave 83.4%
  - 10 instruction(s) over 3% of MFMA idle each (largest 4.4%)
- Repeat 2 (median): MFMA idle 472 of 3544 cycles per iteration (issue 36%, lds: ds_read stall 20%, barrier 13%, memory: buffer_load issue stall 11%); 8/10 MFMA regions fit their shadow yet idle (409 cycles); max barrier skew 39 cycles.
- Hot spots, repeat 2 (exposed cycles per iteration and wave):
  - code index 940: `s_barrier` (s_barrier): 0.0 hole + 20.1 stall + 0.0 issue, 4.3% of MFMA idle
  - code index 562: `s_waitcnt lgkmcnt(7)` (s_waitcnt lgkmcnt): 0.0 hole + 18.8 stall + 0.0 issue, 4.0% of MFMA idle
  - code index 613: `buffer_load_dwordx4 v11, s[0:3], 0 offen lds` (buffer_load): 0.0 hole + 15.9 stall + 0.5 issue, 3.5% of MFMA idle
  - code index 635: `s_barrier` (s_barrier): 0.0 hole + 15.0 stall + 0.0 issue, 3.2% of MFMA idle
  - code index 693: `ds_read_b128 v[98:101], v37` after `v_add_u32_e32 v37, 0x128c0, v10` (ds_read): 8.0 hole + 2.0 stall + 4.0 issue, 3.0% of MFMA idle
  - code index 698: `ds_read_b128 v[78:81], v37` after `v_add_u32_e32 v37, 0x1ac80, v10` (ds_read): 8.0 hole + 2.0 stall + 4.0 issue, 3.0% of MFMA idle
  - wait at code index 562 `s_waitcnt lgkmcnt(7)` (lgkmcnt): stalls 19 cycles 0.98x per iteration, 309 cycles (16 MFMAs) after the load it waits on

#### 256x176 (loop 80.6%, 82.6%, 82.3%; ATT clock 1.73, 1.46, 1.44 GHz)

- Repeat 1: `v9_tile/tmp/tile_n/att_r1/sweep_att/256x176_4096x2816x8192/ui_output_agent_51288_dispatch_64`
  - loop 80.6% vs window ceiling 88% at 1.73 GHz (under by 6.9 points)
  - chip-wide 70.2% < whole wave 76.8%
  - 2 instruction(s) over 3% of MFMA idle each (largest 4.3%)
- Repeat 2: `v9_tile/tmp/tile_n/att_r2/sweep_att/256x176_4096x2816x8192/ui_output_agent_10625_dispatch_64`
  - loop 82.6% vs window ceiling 100% at 1.46 GHz (under by 17.4 points)
  - chip-wide 71.4% < whole wave 78.5%
  - 1 instruction(s) over 3% of MFMA idle each (largest 3.3%)
- Repeat 3: `v9_tile/tmp/tile_n/att_r3/sweep_att/256x176_4096x2816x8192/ui_output_agent_27457_dispatch_64` (wave files se0_sm0_sl0_wv0.json, se0_sm1_sl0_wv0.json, se0_sm2_sl0_wv0.json, se0_sm3_sl0_wv0.json, se2_sm0_sl0_wv0.json, se2_sm1_sl0_wv0.json, se2_sm2_sl0_wv0.json, se2_sm3_sl0_wv0.json, se3_sm0_sl0_wv0.json, se3_sm1_sl0_wv0.json, se3_sm2_sl0_wv0.json, se3_sm3_sl0_wv0.json)
  - loop 82.3% vs window ceiling 100% at 1.44 GHz (under by 17.7 points)
  - chip-wide 71.3% < whole wave 78.2%
  - 1 instruction(s) over 3% of MFMA idle each (largest 3.3%)
- Repeat 3 (median): MFMA idle 607 of 3423 cycles per iteration (issue 43%, lds: ds_read stall 23%, dependency hole 13%, barrier 9%); 8/8 MFMA regions fit their shadow yet idle (552 cycles); max barrier skew 26 cycles.
- Hot spots, repeat 3 (exposed cycles per iteration and wave):
  - code index 572: `s_waitcnt vmcnt(18) lgkmcnt(0)` (s_waitcnt vmcnt+lgkmcnt): 0.0 hole + 19.7 stall + 0.0 issue, 3.3% of MFMA idle
  - code index 580: `ds_read_b128 v[108:111], v3 offset:34528` (ds_read): 0.0 hole + 12.0 stall + 4.0 issue, 2.6% of MFMA idle
  - code index 793: `ds_read_b128 v[108:111], v3 offset:51424` (ds_read): 0.0 hole + 12.0 stall + 4.0 issue, 2.6% of MFMA idle
  - code index 792: `ds_read_b128 v[132:135], v3 offset:51232` (ds_read): 0.0 hole + 11.0 stall + 4.0 issue, 2.5% of MFMA idle
  - code index 579: `ds_read_b128 v[132:135], v3 offset:34336` (ds_read): 0.0 hole + 11.0 stall + 4.0 issue, 2.5% of MFMA idle
  - code index 573: `s_barrier` (s_barrier): 0.0 hole + 14.7 stall + 0.0 issue, 2.4% of MFMA idle
  - wait at code index 572 `s_waitcnt vmcnt(18) lgkmcnt(0)` (vmcnt): stalls 20 cycles 0.98x per iteration, 2298 cycles (124 MFMAs) after the load it waits on
  - wait at code index 572 `s_waitcnt vmcnt(18) lgkmcnt(0)` (lgkmcnt): stalls 20 cycles 0.98x per iteration, 337 cycles (19 MFMAs) after the load it waits on

#### 256x160 (loop 81.5%, 82.8%, 81.6%; ATT clock 1.83, 1.74, 1.75 GHz)

- Repeat 1: `v9_tile/tmp/tile_n/att_r1/sweep_att/256x160_4096x2560x8192/ui_output_agent_2630_dispatch_66`
  - chip-wide 70.3% < whole wave 77.2%
  - 8 instruction(s) over 3% of MFMA idle each (largest 7.2%)
- Repeat 2: `v9_tile/tmp/tile_n/att_r2/sweep_att/256x160_4096x2560x8192/ui_output_agent_26783_dispatch_66`
  - chip-wide 70.7% < whole wave 78.6%
  - 6 instruction(s) over 3% of MFMA idle each (largest 5.9%)
- Repeat 3: `v9_tile/tmp/tile_n/att_r3/sweep_att/256x160_4096x2560x8192/ui_output_agent_44736_dispatch_66` (wave files se0_sm0_sl0_wv0.json, se0_sm1_sl0_wv0.json, se0_sm2_sl0_wv0.json, se0_sm3_sl0_wv0.json, se2_sm0_sl0_wv0.json, se2_sm1_sl0_wv0.json, se2_sm2_sl0_wv0.json, se2_sm3_sl0_wv0.json, se3_sm0_sl0_wv0.json, se3_sm1_sl0_wv0.json, se3_sm2_sl0_wv0.json, se3_sm3_sl0_wv0.json)
  - chip-wide 70.6% < whole wave 77.7%
  - 7 instruction(s) over 3% of MFMA idle each (largest 5.4%)
- Repeat 3 (median): MFMA idle 576 of 3136 cycles per iteration (memory: buffer_load issue stall 32%, issue 30%, barrier 16%, lds: ds_read stall 11%); 8/8 MFMA regions fit their shadow yet idle (487 cycles); max barrier skew 39 cycles.
- Hot spots, repeat 3 (exposed cycles per iteration and wave):
  - code index 863: `buffer_load_dwordx4 v35, s[0:3], 0 offen lds` (buffer_load): 0.0 hole + 30.4 stall + 0.7 issue, 5.4% of MFMA idle
  - code index 578: `buffer_load_dwordx4 v6, s[4:7], 0 offen lds` (buffer_load): 0.0 hole + 24.5 stall + 0.7 issue, 4.4% of MFMA idle
  - code index 523: `s_barrier` (s_barrier): 0.0 hole + 23.4 stall + 0.0 issue, 4.1% of MFMA idle
  - code index 560: `s_barrier` (s_barrier): 0.0 hole + 20.1 stall + 0.0 issue, 3.5% of MFMA idle
  - code index 586: `buffer_load_dwordx4 v7, s[4:7], 0 offen lds` (buffer_load): 0.0 hole + 19.1 stall + 0.8 issue, 3.5% of MFMA idle
  - code index 522: `s_waitcnt vmcnt(16) lgkmcnt(0)` (s_waitcnt vmcnt+lgkmcnt): 0.0 hole + 19.7 stall + 0.0 issue, 3.4% of MFMA idle
  - wait at code index 522 `s_waitcnt vmcnt(16) lgkmcnt(0)` (vmcnt): stalls 20 cycles 0.98x per iteration, 2063 cycles (110 MFMAs) after the load it waits on
  - wait at code index 522 `s_waitcnt vmcnt(16) lgkmcnt(0)` (lgkmcnt): stalls 20 cycles 0.98x per iteration, 352 cycles (16 MFMAs) after the load it waits on

#### 256x144 (loop 75.4%, 74.7%, 73.6%; ATT clock 1.56, 1.70, 1.68 GHz)

- Repeat 1: `v9_tile/tmp/tile_n/att_r1/sweep_att/256x144_4096x2304x8192/ui_output_agent_33242_dispatch_68`
  - loop 75.4% vs window ceiling 86% at 1.56 GHz (under by 10.6 points)
  - chip-wide 65.7% < whole wave 71.7%
- Repeat 2: `v9_tile/tmp/tile_n/att_r2/sweep_att/256x144_4096x2304x8192/ui_output_agent_58889_dispatch_68` (wave files se0_sm0_sl0_wv0.json, se0_sm1_sl0_wv0.json, se0_sm2_sl0_wv0.json, se0_sm3_sl0_wv0.json, se2_sm0_sl0_wv0.json, se2_sm1_sl0_wv0.json, se2_sm2_sl0_wv0.json, se2_sm3_sl0_wv0.json, se3_sm0_sl0_wv0.json, se3_sm1_sl0_wv0.json, se3_sm2_sl0_wv0.json, se3_sm3_sl0_wv0.json)
  - chip-wide 63.9% < whole wave 71.2%
  - 2 instruction(s) over 3% of MFMA idle each (largest 3.2%)
- Repeat 3: `v9_tile/tmp/tile_n/att_r3/sweep_att/256x144_4096x2304x8192/ui_output_agent_60489_dispatch_68`
  - loop 73.6% vs window ceiling 80% at 1.68 GHz (under by 6.3 points)
  - chip-wide 64.0% < whole wave 70.0%
  - 1 instruction(s) over 3% of MFMA idle each (largest 4.4%)
- Repeat 2 (median): MFMA idle 780 of 3084 cycles per iteration (issue 38%, lds: ds_read stall 26%, memory: buffer_load issue stall 11%, barrier 10%); 8/8 MFMA regions fit their shadow yet idle (704 cycles); max barrier skew 35 cycles.
- Hot spots, repeat 2 (exposed cycles per iteration and wave):
  - code index 532: `s_barrier` (s_barrier): 0.0 hole + 24.8 stall + 0.0 issue, 3.2% of MFMA idle
  - code index 875: `buffer_load_dwordx4 v36, s[0:3], 0 offen lds` (buffer_load): 0.0 hole + 23.7 stall + 0.6 issue, 3.1% of MFMA idle
  - code index 531: `s_waitcnt vmcnt(16) lgkmcnt(0)` (s_waitcnt vmcnt+lgkmcnt): 0.0 hole + 19.7 stall + 0.0 issue, 2.5% of MFMA idle
  - code index 673: `ds_read_b128 v[40:43], v16 offset:17664` (ds_read): 0.0 hole + 12.0 stall + 4.0 issue, 2.0% of MFMA idle
  - code index 850: `ds_read_b128 v[56:59], v16 offset:768` (ds_read): 0.0 hole + 12.0 stall + 4.0 issue, 2.0% of MFMA idle
  - code index 539: `ds_read_b128 v[104:107], v16 offset:34528` (ds_read): 0.0 hole + 12.0 stall + 4.0 issue, 2.0% of MFMA idle
  - wait at code index 531 `s_waitcnt vmcnt(16) lgkmcnt(0)` (vmcnt): stalls 20 cycles 0.98x per iteration, 2018 cycles (98 MFMAs) after the load it waits on
  - wait at code index 531 `s_waitcnt vmcnt(16) lgkmcnt(0)` (lgkmcnt): stalls 20 cycles 0.98x per iteration, 316 cycles (15 MFMAs) after the load it waits on

#### 256x128 (loop 71.7%, 70.4%, 76.3%; ATT clock 1.75, 1.83, 1.64 GHz)

- Repeat 1: `v9_tile/tmp/tile_n/att_r1/sweep_att/256x128_4096x2048x8192/ui_output_agent_36579_dispatch_70` (wave files se0_sm0_sl0_wv0.json, se0_sm1_sl0_wv0.json, se0_sm2_sl0_wv0.json, se0_sm3_sl0_wv0.json, se2_sm0_sl0_wv0.json, se2_sm1_sl0_wv0.json, se2_sm2_sl0_wv0.json, se2_sm3_sl0_wv0.json, se3_sm0_sl0_wv0.json, se3_sm1_sl0_wv0.json, se3_sm2_sl0_wv0.json, se3_sm3_sl0_wv0.json)
  - repeats spread 5.9 points (71.7%, 70.4%, 76.3%)
  - chip-wide 62.4% < whole wave 68.6%
  - 6 instruction(s) over 3% of MFMA idle each (largest 6.8%)
- Repeat 2: `v9_tile/tmp/tile_n/att_r2/sweep_att/256x128_4096x2048x8192/ui_output_agent_45373_dispatch_70`
  - chip-wide 60.9% < whole wave 67.4%
  - 6 instruction(s) over 3% of MFMA idle each (largest 5.4%)
- Repeat 3: `v9_tile/tmp/tile_n/att_r3/sweep_att/256x128_4096x2048x8192/ui_output_agent_12426_dispatch_70`
  - chip-wide 65.5% < whole wave 72.8%
  - 6 instruction(s) over 3% of MFMA idle each (largest 6.4%)
- Repeat 1 (median): MFMA idle 808 of 2856 cycles per iteration (memory: buffer_load issue stall 42%, issue 27%, barrier 14%, lds: ds_read stall 10%); 8/8 MFMA regions fit their shadow yet idle (698 cycles); max barrier skew 45 cycles.
- Hot spots, repeat 1 (exposed cycles per iteration and wave):
  - code index 773: `buffer_load_dwordx4 v32, s[0:3], 0 offen lds` (buffer_load): 0.0 hole + 54.3 stall + 1.0 issue, 6.8% of MFMA idle
  - code index 532: `buffer_load_dwordx4 v6, s[4:7], 0 offen lds` (buffer_load): 0.0 hole + 40.1 stall + 1.1 issue, 5.1% of MFMA idle
  - code index 540: `buffer_load_dwordx4 v7, s[4:7], 0 offen lds` (buffer_load): 0.0 hole + 34.7 stall + 1.2 issue, 4.5% of MFMA idle
  - code index 479: `s_barrier` (s_barrier): 0.0 hole + 29.4 stall + 0.0 issue, 3.6% of MFMA idle
  - code index 765: `buffer_load_dwordx4 v31, s[0:3], 0 offen lds` (buffer_load): 0.0 hole + 26.0 stall + 1.8 issue, 3.4% of MFMA idle
  - code index 719: `buffer_load_dwordx4 v28, s[4:7], 0 offen lds` (buffer_load): 0.0 hole + 22.9 stall + 1.5 issue, 3.0% of MFMA idle
  - wait at code index 478 `s_waitcnt vmcnt(14) lgkmcnt(0)` (vmcnt): stalls 20 cycles 0.98x per iteration, 1835 cycles (86 MFMAs) after the load it waits on
  - wait at code index 478 `s_waitcnt vmcnt(14) lgkmcnt(0)` (lgkmcnt): stalls 20 cycles 0.98x per iteration, 289 cycles (11 MFMAs) after the load it waits on

