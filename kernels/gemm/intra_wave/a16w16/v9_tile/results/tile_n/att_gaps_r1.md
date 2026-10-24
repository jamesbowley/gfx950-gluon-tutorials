### 256x128_4096x2048x8192

`v9_tile/tmp/tile_n/att_r1/sweep_att/256x128_4096x2048x8192/ui_output_agent_36579_dispatch_70`: 12 waves, 63 iterations each. Per iteration and wave: 2856 cycles, 808 of them MFMA idle (loop eff 71.7%); MFMA stall 845 pipe-full, 6 pipe-empty; unattributed -4.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| memory: buffer_load issue stall | 341 | 42.2% |
| issue | 218 | 26.9% |
| barrier | 114 | 14.1% |
| lds: ds_read stall | 80 | 9.9% |
| other stall | 28 | 3.5% |
| memory: vmcnt wait | 20 | 2.4% |
| mfma stall, pipe empty | 6 | 0.8% |
| dependency hole | 6 | 0.7% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 773 | `buffer_load_dwordx4 v32, s[0:3], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 54.3 | 1.0 | 6.8% |
| 532 | `buffer_load_dwordx4 v6, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 40.1 | 1.1 | 5.1% |
| 540 | `buffer_load_dwordx4 v7, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 34.7 | 1.2 | 4.5% |
| 479 | `s_barrier` | `s_waitcnt vmcnt(14) lgkmcnt(0)` | s_barrier | 0.0 | 29.4 | 0.0 | 3.6% |
| 765 | `buffer_load_dwordx4 v31, s[0:3], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 26.0 | 1.8 | 3.4% |
| 719 | `buffer_load_dwordx4 v28, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 22.9 | 1.5 | 3.0% |
| 548 | `buffer_load_dwordx4 v8, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 23.3 | 0.9 | 3.0% |
| 727 | `buffer_load_dwordx4 v29, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 22.1 | 1.3 | 2.9% |
| 707 | `s_barrier` | `s_waitcnt vmcnt(16) lgkmcnt(0)` | s_barrier | 0.0 | 22.3 | 0.0 | 2.8% |
| 697 | `buffer_load_dwordx4 v26, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 19.6 | 1.3 | 2.6% |
| 681 | `buffer_load_dwordx4 v24, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 19.1 | 0.9 | 2.5% |
| 478 | `s_waitcnt vmcnt(14) lgkmcnt(0)` | `v_accvgpr_write_b32 a99, v84` | s_waitcnt vmcnt+lgkmcnt | 0.0 | 19.7 | 0.0 | 2.4% |
| 558 | `s_barrier` | `s_waitcnt vmcnt(16) lgkmcnt(0)` | s_barrier | 0.0 | 15.9 | 0.0 | 2.0% |
| 745 | `s_barrier` | `s_waitcnt vmcnt(16) lgkmcnt(0)` | s_barrier | 0.0 | 15.5 | 0.0 | 1.9% |
| 527 | `buffer_load_dwordx4 v5, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 11.5 | 4.0 | 1.9% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 479 | 479 `s_barrier` | 0 | 1 | 0 | 0 | yes | 29 | 29 |
| 480 | 515 `s_waitcnt vmcnt(14) lgkmcnt(0)` | 16 | 20 | 184 | 57 | yes | 39 | 4 |
| 516 | 516 `s_barrier` | 0 | 1 | 0 | 0 | yes | 12 | 20 |
| 517 | 557 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 16 | 25 | 183 | 70 | yes | 173 | 4 |
| 558 | 558 `s_barrier` | 0 | 1 | 0 | 0 | yes | 16 | 24 |
| 559 | 595 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 16 | 21 | 183 | 54 | yes | 68 | 4 |
| 596 | 596 `s_barrier` | 0 | 1 | 0 | 0 | yes | 9 | 17 |
| 597 | 633 `s_waitcnt vmcnt(14) lgkmcnt(0)` | 16 | 21 | 180 | 55 | yes | 29 | 4 |
| 635 | 668 `s_waitcnt vmcnt(14) lgkmcnt(0)` | 16 | 18 | 184 | 49 | yes | 24 | 4 |
| 670 | 706 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 16 | 21 | 183 | 54 | yes | 107 | 4 |
| 707 | 707 `s_barrier` | 0 | 1 | 0 | 0 | yes | 22 | 30 |
| 708 | 744 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 16 | 21 | 183 | 54 | yes | 109 | 4 |
| 745 | 745 `s_barrier` | 0 | 1 | 0 | 0 | yes | 16 | 24 |
| 746 | 478 `s_waitcnt vmcnt(14) lgkmcnt(0)` | 16 | 28 | 180 | 83 | yes | 150 | 20 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 478 | `s_waitcnt vmcnt(14) lgkmcnt(0)` | vmcnt | 0.98 | 20 | 1835 | 86.0 |
| 478 | `s_waitcnt vmcnt(14) lgkmcnt(0)` | lgkmcnt | 0.98 | 20 | 289 | 11.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 479 | 45 | {3: 97, 1: 92} |
| 516 | 27 | {1: 170, 3: 19} |
| 558 | 36 | {1: 115, 3: 73, 2: 1} |
| 596 | 23 | {3: 96, 1: 93} |
| 634 | 13 | {3: 124, 1: 65} |
| 669 | 13 | {3: 125, 1: 64} |
| 707 | 45 | {1: 127, 3: 62} |
| 745 | 36 | {1: 117, 3: 72} |

### 256x144_4096x2304x8192

`v9_tile/tmp/tile_n/att_r1/sweep_att/256x144_4096x2304x8192/ui_output_agent_33242_dispatch_68`: 12 waves, 63 iterations each. Per iteration and wave: 3057 cycles, 753 of them MFMA idle (loop eff 75.4%); MFMA stall 922 pipe-full, 1 pipe-empty; unattributed -0.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| issue | 298 | 39.6% |
| lds: ds_read stall | 203 | 26.9% |
| barrier | 74 | 9.9% |
| memory: buffer_load issue stall | 64 | 8.6% |
| dependency hole | 63 | 8.4% |
| other stall | 29 | 3.9% |
| memory: vmcnt wait | 20 | 2.6% |
| mfma stall, pipe empty | 1 | 0.2% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 532 | `s_barrier` | `s_waitcnt vmcnt(16) lgkmcnt(0)` | s_barrier | 0.0 | 22.6 | 0.0 | 3.0% |
| 531 | `s_waitcnt vmcnt(16) lgkmcnt(0)` | `v_accvgpr_write_b32 a115, v0` | s_waitcnt vmcnt+lgkmcnt | 0.0 | 19.7 | 0.0 | 2.6% |
| 573 | `s_barrier` | `s_waitcnt vmcnt(15) lgkmcnt(0)` | s_barrier | 0.0 | 16.2 | 0.0 | 2.2% |
| 673 | `ds_read_b128 v[40:43], v16 offset:17664` | `ds_read_b128 v[60:63], v16 offset:17472` | ds_read | 0.0 | 12.0 | 4.0 | 2.1% |
| 850 | `ds_read_b128 v[56:59], v16 offset:768` | `ds_read_b128 v[64:67], v16 offset:576` | ds_read | 0.0 | 11.9 | 4.0 | 2.1% |
| 539 | `ds_read_b128 v[104:107], v16 offset:34528` | `ds_read_b128 v[132:135], v16 offset:3433` | ds_read | 0.0 | 11.9 | 4.0 | 2.1% |
| 538 | `ds_read_b128 v[132:135], v16 offset:34336` | `ds_read_b128 v[128:131], v16 offset:3427` | ds_read | 0.0 | 10.9 | 4.1 | 2.0% |
| 672 | `ds_read_b128 v[60:63], v16 offset:17472` | `ds_read_b128 v[56:59], v16 offset:17408` | ds_read | 0.0 | 11.0 | 4.0 | 2.0% |
| 849 | `ds_read_b128 v[64:67], v16 offset:576` | `ds_read_b128 v[68:71], v16 offset:512` | ds_read | 0.0 | 10.9 | 4.1 | 2.0% |
| 722 | `ds_read_b128 v[132:135], v16 offset:51232` | `ds_read_b128 v[128:131], v16 offset:5116` | ds_read | 0.0 | 10.0 | 5.0 | 2.0% |
| 723 | `ds_read_b128 v[104:107], v16 offset:51424` | `ds_read_b128 v[132:135], v16 offset:5123` | ds_read | 0.0 | 11.0 | 4.0 | 2.0% |
| 800 | `ds_read_b128 v[92:95], v0` | `v_add_u32_e32 v0, 0x107c0, v39` | ds_read | 7.9 | 2.0 | 4.1 | 1.9% |
| 623 | `ds_read_b128 v[76:79], v0` | `v_add_u32_e32 v0, 0x128c0, v39` | ds_read | 8.0 | 2.0 | 4.0 | 1.9% |
| 576 | `ds_read_b128 v[156:159], v0` | `v_add_u32_e32 v0, 0x149a0, v39` | ds_read | 7.0 | 2.0 | 4.0 | 1.7% |
| 757 | `ds_read_b128 v[156:159], v0` | `v_add_u32_e32 v0, 0x16aa0, v39` | ds_read | 6.0 | 2.0 | 5.0 | 1.7% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 532 | 532 `s_barrier` | 0 | 1 | 0 | 0 | yes | 23 | 23 |
| 533 | 572 `s_waitcnt vmcnt(15) lgkmcnt(0)` | 16 | 24 | 185 | 73 | yes | 85 | 4 |
| 573 | 573 `s_barrier` | 0 | 1 | 0 | 0 | yes | 16 | 24 |
| 574 | 620 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 16 | 31 | 185 | 92 | yes | 137 | 4 |
| 621 | 621 `s_barrier` | 0 | 1 | 0 | 0 | yes | 11 | 19 |
| 622 | 665 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 20 | 24 | 232 | 55 | yes | 42 | 4 |
| 667 | 715 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 20 | 29 | 232 | 74 | yes | 75 | 4 |
| 716 | 716 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 717 | 754 `s_waitcnt vmcnt(15) lgkmcnt(0)` | 16 | 22 | 184 | 71 | yes | 71 | 4 |
| 755 | 755 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 756 | 797 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 16 | 26 | 185 | 78 | yes | 89 | 4 |
| 799 | 842 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 20 | 24 | 232 | 55 | yes | 41 | 4 |
| 844 | 531 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 20 | 36 | 232 | 103 | yes | 139 | 20 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 531 | `s_waitcnt vmcnt(16) lgkmcnt(0)` | vmcnt | 0.98 | 20 | 1991 | 98.0 |
| 531 | `s_waitcnt vmcnt(16) lgkmcnt(0)` | lgkmcnt | 0.98 | 20 | 294 | 15.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 532 | 32 | {1: 136, 3: 52, 2: 1} |
| 573 | 33 | {1: 137, 3: 52} |
| 621 | 24 | {1: 132, 3: 51, 2: 4, 0: 2} |
| 666 | 13 | {1: 135, 3: 53, 0: 1} |
| 716 | 14 | {1: 135, 3: 54} |
| 755 | 14 | {1: 135, 3: 54} |
| 798 | 12 | {1: 135, 3: 54} |
| 843 | 13 | {1: 135, 3: 54} |

### 256x160_4096x2560x8192

`v9_tile/tmp/tile_n/att_r1/sweep_att/256x160_4096x2560x8192/ui_output_agent_2630_dispatch_66`: 12 waves, 63 iterations each. Per iteration and wave: 3139 cycles, 579 of them MFMA idle (loop eff 81.5%); MFMA stall 1017 pipe-full, 3 pipe-empty; unattributed -1.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| memory: buffer_load issue stall | 193 | 33.3% |
| issue | 171 | 29.6% |
| barrier | 84 | 14.4% |
| lds: ds_read stall | 61 | 10.5% |
| dependency hole | 25 | 4.3% |
| other stall | 24 | 4.1% |
| memory: vmcnt wait | 20 | 3.4% |
| mfma stall, pipe empty | 3 | 0.5% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 863 | `buffer_load_dwordx4 v35, s[0:3], 0 offen lds` | `v_mfma_f32_16x16x32_bf16 a[20:23], v[128` | buffer_load | 0.0 | 41.0 | 0.8 | 7.2% |
| 578 | `buffer_load_dwordx4 v6, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 35.5 | 0.9 | 6.3% |
| 586 | `buffer_load_dwordx4 v7, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 26.8 | 0.9 | 4.8% |
| 523 | `s_barrier` | `s_waitcnt vmcnt(16) lgkmcnt(0)` | s_barrier | 0.0 | 26.4 | 0.0 | 4.6% |
| 871 | `buffer_load_dwordx4 v36, s[0:3], 0 offen lds` | `v_mfma_f32_16x16x32_bf16 a[12:15], v[144` | buffer_load | 0.0 | 20.2 | 0.6 | 3.6% |
| 522 | `s_waitcnt vmcnt(16) lgkmcnt(0)` | `v_accvgpr_write_b32 a131, v39` | s_waitcnt vmcnt+lgkmcnt | 0.0 | 19.7 | 0.0 | 3.4% |
| 560 | `s_barrier` | `s_waitcnt vmcnt(15) lgkmcnt(0)` | s_barrier | 0.0 | 18.5 | 0.0 | 3.2% |
| 573 | `buffer_load_dwordx4 v5, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 14.1 | 4.0 | 3.1% |
| 855 | `buffer_load_dwordx4 v34, s[0:3], 0 offen lds` | `v_mfma_f32_16x16x32_bf16 a[28:31], v[128` | buffer_load | 0.0 | 13.9 | 0.6 | 2.5% |
| 566 | `ds_read_b128 v[124:127], v37 offset:64` | `ds_read_b128 v[120:123], v37` | ds_read | 0.0 | 10.0 | 4.0 | 2.4% |
| 784 | `ds_read_b128 v[80:83], v39` | `v_add_u32_e32 v39, 0x107c0, v16` | ds_read | 8.0 | 2.0 | 4.0 | 2.4% |
| 594 | `buffer_load_dwordx4 v8, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 13.3 | 0.5 | 2.4% |
| 741 | `ds_read_b128 v[124:127], v39` | `v_add_u32_e32 v39, 0x16aa0, v16` | ds_read | 6.0 | 2.0 | 5.0 | 2.2% |
| 746 | `ds_read_b128 v[140:143], v39` | `v_add_u32_e32 v39, 0x19c00, v16` | ds_read | 7.0 | 2.0 | 4.0 | 2.2% |
| 605 | `s_barrier` | `v_mfma_f32_16x16x32_bf16 a[96:99], v[132` | s_barrier | 0.0 | 9.7 | 0.0 | 1.7% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 523 | 523 `s_barrier` | 0 | 1 | 0 | 0 | yes | 26 | 26 |
| 524 | 559 `s_waitcnt vmcnt(15) lgkmcnt(0)` | 16 | 20 | 185 | 57 | yes | 42 | 4 |
| 560 | 560 `s_barrier` | 0 | 1 | 0 | 0 | yes | 18 | 26 |
| 561 | 603 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 16 | 27 | 183 | 80 | yes | 175 | 4 |
| 604 | 605 `s_barrier` | 1 | 1 | 12 | 0 | yes | 10 | 20 |
| 606 | 652 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 23 | 24 | 275 | 48 | yes | 8 | 4 |
| 655 | 703 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 23 | 26 | 276 | 56 | yes | 7 | 4 |
| 705 | 738 `s_waitcnt vmcnt(15) lgkmcnt(0)` | 16 | 18 | 185 | 49 | yes | 26 | 4 |
| 739 | 739 `s_barrier` | 0 | 1 | 0 | 0 | yes | 7 | 15 |
| 740 | 780 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 16 | 25 | 185 | 73 | yes | 83 | 4 |
| 781 | 782 `s_barrier` | 1 | 1 | 12 | 0 | yes | 6 | 17 |
| 783 | 830 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 23 | 25 | 276 | 52 | yes | 30 | 4 |
| 831 | 832 `s_barrier` | 1 | 1 | 12 | 0 | yes | 8 | 18 |
| 833 | 522 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 23 | 33 | 276 | 84 | yes | 126 | 20 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 522 | `s_waitcnt vmcnt(16) lgkmcnt(0)` | vmcnt | 0.98 | 20 | 2050 | 110.0 |
| 522 | `s_waitcnt vmcnt(16) lgkmcnt(0)` | lgkmcnt | 0.98 | 20 | 367 | 16.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 523 | 39 | {3: 152, 1: 37} |
| 560 | 35 | {3: 169, 1: 20} |
| 605 | 27 | {3: 151, 1: 38} |
| 654 | 13 | {3: 183, 1: 6} |
| 704 | 12 | {3: 188, 1: 1} |
| 739 | 16 | {3: 181, 1: 8} |
| 782 | 19 | {3: 171, 1: 17, 2: 1} |
| 832 | 22 | {3: 158, 1: 30, 2: 1} |

### 256x176_4096x2816x8192

`v9_tile/tmp/tile_n/att_r1/sweep_att/256x176_4096x2816x8192/ui_output_agent_51288_dispatch_64`: 12 waves, 63 iterations each. Per iteration and wave: 3492 cycles, 676 of them MFMA idle (loop eff 80.6%); MFMA stall 1100 pipe-full, 3 pipe-empty; unattributed 0.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| issue | 262 | 38.8% |
| lds: ds_read stall | 139 | 20.6% |
| dependency hole | 80 | 11.8% |
| memory: buffer_load issue stall | 76 | 11.3% |
| barrier | 70 | 10.4% |
| other stall | 26 | 3.8% |
| memory: vmcnt wait | 20 | 2.9% |
| mfma stall, pipe empty | 3 | 0.5% |
| lds: lgkmcnt wait | 0 | 0.0% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 604 | `buffer_load_dwordx4 v6, s[0:3], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 28.2 | 0.6 | 4.3% |
| 614 | `s_barrier` | `s_waitcnt vmcnt(16) lgkmcnt(0)` | s_barrier | 0.0 | 20.7 | 0.0 | 3.1% |
| 572 | `s_waitcnt vmcnt(18) lgkmcnt(0)` | `v_accvgpr_write_b32 a147, v0` | s_waitcnt vmcnt+lgkmcnt | 0.0 | 19.7 | 0.0 | 2.9% |
| 573 | `s_barrier` | `s_waitcnt vmcnt(18) lgkmcnt(0)` | s_barrier | 0.0 | 19.4 | 0.0 | 2.9% |
| 580 | `ds_read_b128 v[108:111], v3 offset:34528` | `ds_read_b128 v[132:135], v3 offset:34336` | ds_read | 0.0 | 12.0 | 4.0 | 2.4% |
| 793 | `ds_read_b128 v[108:111], v3 offset:51424` | `ds_read_b128 v[132:135], v3 offset:51232` | ds_read | 0.0 | 12.0 | 4.0 | 2.4% |
| 579 | `ds_read_b128 v[132:135], v3 offset:34336` | `ds_read_b128 v[124:127], v3 offset:34272` | ds_read | 0.0 | 11.0 | 4.0 | 2.2% |
| 792 | `ds_read_b128 v[132:135], v3 offset:51232` | `ds_read_b128 v[124:127], v3 offset:51168` | ds_read | 0.0 | 11.0 | 4.0 | 2.2% |
| 672 | `ds_read_b128 v[92:95], v0` | `v_add_u32_e32 v0, 0x128c0, v21` | ds_read | 8.0 | 2.0 | 4.0 | 2.1% |
| 878 | `ds_read_b128 v[96:99], v0` | `v_add_u32_e32 v0, 0x107c0, v21` | ds_read | 8.0 | 2.0 | 4.0 | 2.1% |
| 622 | `ds_read_b128 v[164:167], v0` | `v_add_u32_e32 v0, 0x18b80, v21` | ds_read | 7.9 | 2.0 | 4.0 | 2.1% |
| 832 | `ds_read_b128 v[176:179], v0` | `v_add_u32_e32 v0, 0x19c00, v21` | ds_read | 7.0 | 2.0 | 4.3 | 2.0% |
| 617 | `ds_read_b128 v[180:183], v0` | `v_add_u32_e32 v0, 0x149a0, v21` | ds_read | 7.3 | 2.0 | 4.1 | 2.0% |
| 625 | `ds_read_b128 v[156:159], v0` | `v_add_u32_e32 v0, 0x1ac60, v20` | ds_read | 8.0 | 1.1 | 4.0 | 1.9% |
| 827 | `ds_read_b128 v[160:163], v0` | `v_add_u32_e32 v0, 0x16aa0, v21` | ds_read | 6.0 | 2.0 | 5.0 | 1.9% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 573 | 573 `s_barrier` | 0 | 1 | 0 | 0 | yes | 19 | 19 |
| 574 | 613 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 16 | 24 | 185 | 73 | yes | 114 | 4 |
| 614 | 614 `s_barrier` | 0 | 1 | 0 | 0 | yes | 21 | 29 |
| 615 | 664 `s_waitcnt lgkmcnt(7)` | 16 | 34 | 185 | 104 | yes | 148 | 4 |
| 669 | 670 `s_barrier` | 1 | 1 | 12 | 0 | yes | 5 | 16 |
| 671 | 721 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 25 | 26 | 300 | 52 | yes | 20 | 4 |
| 722 | 722 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 723 | 785 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 28 | 35 | 328 | 83 | yes | 29 | 4 |
| 786 | 786 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 787 | 824 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 16 | 22 | 185 | 65 | yes | 70 | 4 |
| 825 | 825 `s_barrier` | 0 | 1 | 0 | 0 | yes | 6 | 14 |
| 826 | 870 `s_waitcnt lgkmcnt(7)` | 16 | 29 | 185 | 90 | yes | 111 | 4 |
| 877 | 927 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 25 | 26 | 300 | 52 | yes | 21 | 4 |
| 928 | 928 `s_barrier` | 0 | 1 | 0 | 0 | yes | 6 | 14 |
| 929 | 572 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 28 | 42 | 328 | 111 | yes | 92 | 20 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 572 | `s_waitcnt vmcnt(18) lgkmcnt(0)` | vmcnt | 0.98 | 20 | 2321 | 124.0 |
| 572 | `s_waitcnt vmcnt(18) lgkmcnt(0)` | lgkmcnt | 0.98 | 20 | 358 | 19.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 573 | 25 | {3: 128, 1: 60, 2: 1} |
| 614 | 42 | {3: 129, 1: 59, 2: 1} |
| 670 | 19 | {3: 129, 1: 60} |
| 722 | 13 | {3: 129, 1: 60} |
| 786 | 13 | {3: 129, 1: 60} |
| 825 | 14 | {3: 129, 1: 60} |
| 876 | 14 | {3: 129, 1: 60} |
| 928 | 14 | {3: 129, 1: 60} |

### 256x192_4096x3072x8192

`v9_tile/tmp/tile_n/att_r1/sweep_att/256x192_4096x3072x8192/ui_output_agent_64795_dispatch_64`: 12 waves, 63 iterations each. Per iteration and wave: 3562 cycles, 490 of them MFMA idle (loop eff 86.2%); MFMA stall 1302 pipe-full, 6 pipe-empty; unattributed 0.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| issue | 171 | 34.8% |
| lds: ds_read stall | 95 | 19.4% |
| memory: buffer_load issue stall | 72 | 14.7% |
| barrier | 60 | 12.2% |
| dependency hole | 43 | 8.8% |
| other stall | 25 | 5.1% |
| lds: lgkmcnt wait | 19 | 3.8% |
| mfma stall, pipe empty | 6 | 1.2% |
| memory: vmcnt wait | 0 | 0.0% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 613 | `buffer_load_dwordx4 v11, s[0:3], 0 offen lds` | `v_mfma_f32_16x16x32_bf16 a[140:143], v[6` | buffer_load | 0.0 | 29.7 | 0.7 | 6.2% |
| 968 | `buffer_load_dwordx4 v35, s[0:3], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 18.6 | 0.6 | 3.9% |
| 562 | `s_waitcnt lgkmcnt(7)` | `v_accvgpr_write_b32 a163, v37` | s_waitcnt lgkmcnt | 0.0 | 18.8 | 0.0 | 3.8% |
| 635 | `s_barrier` | `v_mfma_f32_16x16x32_bf16 a[128:131], v[9` | s_barrier | 0.0 | 17.7 | 0.0 | 3.6% |
| 693 | `ds_read_b128 v[98:101], v37` | `v_add_u32_e32 v37, 0x128c0, v10` | ds_read | 8.0 | 2.0 | 4.0 | 2.9% |
| 698 | `ds_read_b128 v[78:81], v37` | `v_add_u32_e32 v37, 0x1ac80, v10` | ds_read | 8.0 | 2.0 | 4.0 | 2.9% |
| 845 | `ds_read_b128 v[138:141], v37` | `v_add_u32_e32 v37, 0x16aa0, v10` | ds_read | 8.0 | 2.0 | 4.0 | 2.9% |
| 898 | `ds_read_b128 v[98:101], v37` | `v_add_u32_e32 v37, 0x107c0, v10` | ds_read | 8.0 | 2.0 | 4.0 | 2.9% |
| 903 | `ds_read_b128 v[78:81], v37` | `v_add_u32_e32 v37, 0x18b80, v10` | ds_read | 8.0 | 2.0 | 4.0 | 2.9% |
| 568 | `s_barrier` | `v_mfma_f32_16x16x32_bf16 a[184:187], v[8` | s_barrier | 0.0 | 13.3 | 0.0 | 2.7% |
| 700 | `ds_read_b128 v[62:65], v37 offset:4224` | `ds_read_b128 v[82:85], v37 offset:64` | ds_read | 0.0 | 9.0 | 4.0 | 2.7% |
| 905 | `ds_read_b128 v[62:65], v37 offset:4224` | `ds_read_b128 v[82:85], v37 offset:64` | ds_read | 0.0 | 9.0 | 4.0 | 2.7% |
| 960 | `buffer_load_dwordx4 v34, s[0:3], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 11.0 | 1.6 | 2.6% |
| 701 | `ds_read_b128 v[66:69], v37 offset:4288` | `ds_read_b128 v[62:65], v37 offset:4224` | ds_read | 0.0 | 8.0 | 4.0 | 2.4% |
| 906 | `ds_read_b128 v[66:69], v37 offset:4288` | `ds_read_b128 v[62:65], v37 offset:4224` | ds_read | 0.0 | 8.0 | 4.0 | 2.4% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 567 | 568 `s_barrier` | 1 | 1 | 12 | 0 | yes | 13 | 24 |
| 569 | 626 `s_waitcnt lgkmcnt(7)` | 29 | 29 | 348 | 72 | yes | 48 | 4 |
| 629 | 633 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 3 | 2 | 36 | 0 | yes | 0 | 4 |
| 634 | 635 `s_barrier` | 1 | 1 | 12 | 0 | yes | 18 | 28 |
| 636 | 690 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 27 | 28 | 324 | 64 | yes | 18 | 4 |
| 691 | 691 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 692 | 734 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 16 | 27 | 184 | 75 | yes | 107 | 4 |
| 735 | 735 `s_barrier` | 0 | 1 | 0 | 0 | yes | 8 | 16 |
| 736 | 772 `s_waitcnt lgkmcnt(7)` | 16 | 21 | 184 | 51 | yes | 28 | 4 |
| 779 | 834 `s_waitcnt lgkmcnt(7)` | 29 | 27 | 348 | 64 | yes | 6 | 4 |
| 837 | 841 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 3 | 2 | 36 | 0 | yes | 0 | 4 |
| 844 | 895 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 27 | 25 | 324 | 52 | yes | 18 | 4 |
| 897 | 939 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 16 | 27 | 184 | 75 | yes | 107 | 4 |
| 940 | 940 `s_barrier` | 0 | 1 | 0 | 0 | yes | 8 | 16 |
| 941 | 562 `s_waitcnt lgkmcnt(7)` | 16 | 28 | 184 | 81 | yes | 98 | 19 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 562 | `s_waitcnt lgkmcnt(7)` | lgkmcnt | 0.98 | 19 | 334 | 16.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 568 | 33 | {1: 140, 3: 49} |
| 635 | 43 | {1: 143, 3: 46} |
| 691 | 13 | {1: 131, 3: 58} |
| 735 | 22 | {1: 142, 3: 47} |
| 778 | 11 | {1: 126, 3: 63} |
| 843 | 11 | {1: 126, 3: 63} |
| 896 | 11 | {1: 126, 3: 63} |
| 940 | 22 | {1: 143, 3: 46} |

### 256x208_4096x3328x8192

`v9_tile/tmp/tile_n/att_r1/sweep_att/256x208_4096x3328x8192/ui_output_agent_10815_dispatch_62`: 12 waves, 63 iterations each. Per iteration and wave: 3917 cycles, 589 of them MFMA idle (loop eff 85.0%); MFMA stall 1359 pipe-full, 1 pipe-empty; unattributed 0.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| issue | 206 | 34.9% |
| lds: ds_read stall | 150 | 25.5% |
| dependency hole | 80 | 13.5% |
| barrier | 67 | 11.4% |
| memory: buffer_load issue stall | 38 | 6.5% |
| other stall | 27 | 4.6% |
| memory: vmcnt wait | 20 | 3.4% |
| mfma stall, pipe empty | 1 | 0.2% |
| lds: lgkmcnt wait | 0 | 0.0% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 616 | `s_waitcnt vmcnt(18) lgkmcnt(0)` | `v_accvgpr_write_b32 a179, v0` | s_waitcnt vmcnt+lgkmcnt | 0.0 | 19.7 | 0.0 | 3.4% |
| 671 | `buffer_load_dwordx4 v16, s[0:3], 0 offen lds` | `v_mfma_f32_16x16x32_bf16 a[156:159], v[1` | buffer_load | 0.0 | 15.8 | 0.6 | 2.8% |
| 809 | `ds_read_b128 v[44:47], v1 offset:17664` | `ds_read_b128 v[64:67], v1 offset:17472` | ds_read | 0.0 | 12.0 | 4.0 | 2.7% |
| 1037 | `ds_read_b128 v[60:63], v1 offset:768` | `ds_read_b128 v[72:75], v1 offset:576` | ds_read | 0.0 | 12.0 | 4.0 | 2.7% |
| 808 | `ds_read_b128 v[64:67], v1 offset:17472` | `ds_read_b128 v[60:63], v1 offset:17408` | ds_read | 0.0 | 11.0 | 4.0 | 2.5% |
| 1036 | `ds_read_b128 v[72:75], v1 offset:576` | `ds_read_b128 v[68:71], v1 offset:512` | ds_read | 0.0 | 11.0 | 4.0 | 2.5% |
| 754 | `ds_read_b128 v[116:119], v0` | `v_add_u32_e32 v0, 0x128c0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 2.4% |
| 759 | `ds_read_b128 v[88:91], v0` | `v_add_u32_e32 v0, 0x1ac80, v12` | ds_read | 8.0 | 2.0 | 4.0 | 2.4% |
| 924 | `ds_read_b128 v[176:179], v0` | `v_add_u32_e32 v0, 0x16aa0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 2.4% |
| 982 | `ds_read_b128 v[116:119], v0` | `v_add_u32_e32 v0, 0x107c0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 2.4% |
| 987 | `ds_read_b128 v[100:103], v0` | `v_add_u32_e32 v0, 0x18b80, v12` | ds_read | 8.0 | 2.0 | 4.0 | 2.4% |
| 1030 | `s_barrier` | `s_waitcnt vmcnt(19) lgkmcnt(0)` | s_barrier | 0.0 | 14.0 | 0.0 | 2.4% |
| 692 | `ds_read_b128 v[176:179], v0` | `v_add_u32_e32 v0, 0x149a0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 2.4% |
| 802 | `s_barrier` | `s_waitcnt vmcnt(19) lgkmcnt(0)` | s_barrier | 0.0 | 14.0 | 0.0 | 2.4% |
| 689 | `s_barrier` | `s_waitcnt vmcnt(19) lgkmcnt(0)` | s_barrier | 0.0 | 13.8 | 0.0 | 2.3% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 617 | 618 `s_barrier` | 1 | 1 | 12 | 0 | yes | 16 | 16 |
| 619 | 681 `s_waitcnt lgkmcnt(11)` | 31 | 32 | 372 | 88 | yes | 41 | 4 |
| 684 | 688 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 3 | 2 | 36 | 0 | yes | 0 | 4 |
| 689 | 689 `s_barrier` | 0 | 1 | 0 | 0 | yes | 14 | 22 |
| 690 | 751 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 28 | 34 | 328 | 91 | yes | 58 | 4 |
| 752 | 752 `s_barrier` | 0 | 1 | 0 | 0 | yes | 6 | 14 |
| 753 | 801 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 20 | 29 | 232 | 75 | yes | 86 | 4 |
| 802 | 802 `s_barrier` | 0 | 1 | 0 | 0 | yes | 14 | 22 |
| 803 | 851 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 20 | 29 | 231 | 74 | yes | 74 | 4 |
| 854 | 914 `s_waitcnt lgkmcnt(11)` | 31 | 30 | 372 | 80 | yes | 9 | 4 |
| 917 | 921 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 3 | 2 | 36 | 0 | yes | 0 | 4 |
| 922 | 922 `s_barrier` | 0 | 1 | 0 | 0 | yes | 6 | 14 |
| 923 | 979 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 28 | 29 | 328 | 71 | yes | 38 | 4 |
| 980 | 980 `s_barrier` | 0 | 1 | 0 | 0 | yes | 6 | 14 |
| 981 | 1029 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 20 | 29 | 232 | 75 | yes | 86 | 4 |
| 1030 | 1030 `s_barrier` | 0 | 1 | 0 | 0 | yes | 14 | 22 |
| 1031 | 616 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 20 | 36 | 231 | 102 | yes | 117 | 20 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 616 | `s_waitcnt vmcnt(18) lgkmcnt(0)` | vmcnt | 0.98 | 20 | 2432 | 130.0 |
| 616 | `s_waitcnt vmcnt(18) lgkmcnt(0)` | lgkmcnt | 0.98 | 20 | 272 | 15.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 618 | 18 | {3: 188, 2: 1} |
| 689 | 31 | {3: 187, 2: 2} |
| 752 | 15 | {3: 187, 1: 2} |
| 802 | 31 | {3: 189} |
| 853 | 15 | {3: 189} |
| 922 | 15 | {3: 189} |
| 980 | 15 | {3: 189} |
| 1030 | 31 | {3: 189} |

### 256x224_4096x3584x8192

`v9_tile/tmp/tile_n/att_r1/sweep_att/256x224_4096x3584x8192/ui_output_agent_37173_dispatch_60`: 12 waves, 63 iterations each. Per iteration and wave: 3891 cycles, 307 of them MFMA idle (loop eff 92.1%); MFMA stall 1452 pipe-full, 1 pipe-empty; unattributed 0.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| issue | 119 | 38.6% |
| dependency hole | 55 | 18.0% |
| barrier | 36 | 11.7% |
| other stall | 33 | 10.9% |
| lds: ds_read stall | 28 | 9.0% |
| lds: lgkmcnt wait | 20 | 6.4% |
| memory: buffer_load issue stall | 16 | 5.1% |
| mfma stall, pipe empty | 1 | 0.3% |
| memory: vmcnt wait | 0 | 0.0% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 604 | `s_waitcnt lgkmcnt(7)` | `v_accvgpr_write_b32 a195, v12` | s_waitcnt lgkmcnt | 0.0 | 19.7 | 0.0 | 6.4% |
| 679 | `ds_read_b128 v[146:149], v41` | `v_add_u32_e32 v41, 0x149a0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 4.6% |
| 740 | `ds_read_b128 v[94:97], v41` | `v_add_u32_e32 v41, 0x128c0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 4.6% |
| 916 | `ds_read_b128 v[150:153], v41` | `v_add_u32_e32 v41, 0x16aa0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 4.6% |
| 973 | `ds_read_b128 v[98:101], v41` | `v_add_u32_e32 v41, 0x107c0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 4.6% |
| 981 | `ds_read_b128 v[82:85], v41` | `v_add_u32_e32 v41, 0x18b80, v12` | ds_read | 6.0 | 0.0 | 4.0 | 3.3% |
| 748 | `ds_read_b128 v[74:77], v41` | `v_add_u32_e32 v41, 0x1ac80, v12` | ds_read | 5.5 | 0.1 | 4.0 | 3.1% |
| 676 | `s_barrier` | `s_waitcnt vmcnt(19) lgkmcnt(0)` | s_barrier | 0.0 | 9.0 | 0.0 | 2.9% |
| 688 | `ds_read_b128 v[138:141], v41` | `v_add_u32_e32 v41, 0x1cd60, v12` | ds_read | 4.0 | 0.0 | 4.0 | 2.6% |
| 741 | `ds_read_b128 v[98:101], v41 offset:64` | `ds_read_b128 v[94:97], v41` | ds_read | 0.0 | 4.0 | 4.0 | 2.6% |
| 742 | `ds_read_b128 v[90:93], v41 offset:4224` | `ds_read_b128 v[98:101], v41 offset:64` | ds_read | 0.0 | 4.0 | 4.0 | 2.6% |
| 925 | `ds_read_b128 v[138:141], v41` | `v_add_u32_e32 v41, 0x1dde0, v12` | ds_read | 4.0 | 0.0 | 4.0 | 2.6% |
| 974 | `ds_read_b128 v[102:105], v41 offset:64` | `ds_read_b128 v[98:101], v41` | ds_read | 0.0 | 4.0 | 4.0 | 2.6% |
| 975 | `ds_read_b128 v[90:93], v41 offset:4224` | `ds_read_b128 v[102:105], v41 offset:64` | ds_read | 0.0 | 4.0 | 4.0 | 2.6% |
| 1083 | `s_cbranch_scc1 64730` | `s_cmpk_lt_u32 s18, 0x7c` | other | 4.0 | 0.0 | 4.0 | 2.6% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 611 | 668 `s_waitcnt lgkmcnt(7)` | 29 | 29 | 348 | 72 | yes | 19 | 4 |
| 671 | 675 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 3 | 2 | 36 | 0 | yes | 0 | 4 |
| 676 | 676 `s_barrier` | 0 | 1 | 0 | 0 | yes | 9 | 17 |
| 677 | 737 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 28 | 33 | 328 | 87 | yes | 50 | 4 |
| 738 | 738 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 739 | 793 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 24 | 31 | 280 | 75 | yes | 55 | 4 |
| 796 | 844 `s_waitcnt lgkmcnt(7)` | 23 | 26 | 276 | 56 | yes | 7 | 4 |
| 851 | 906 `s_waitcnt lgkmcnt(7)` | 29 | 27 | 348 | 64 | yes | 7 | 4 |
| 909 | 913 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 3 | 2 | 36 | 0 | yes | 0 | 4 |
| 915 | 970 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 28 | 28 | 328 | 67 | yes | 29 | 4 |
| 971 | 971 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 972 | 1026 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 24 | 31 | 280 | 75 | yes | 56 | 4 |
| 1029 | 604 `s_waitcnt lgkmcnt(7)` | 23 | 33 | 276 | 84 | yes | 47 | 20 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 604 | `s_waitcnt lgkmcnt(7)` | lgkmcnt | 0.98 | 20 | 395 | 23.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 610 | 14 | {1: 96, 3: 93} |
| 676 | 20 | {1: 96, 3: 93} |
| 738 | 13 | {1: 96, 3: 93} |
| 795 | 15 | {1: 49, 0: 48, 2: 47, 3: 45} |
| 850 | 12 | {1: 96, 3: 93} |
| 914 | 13 | {1: 96, 3: 93} |
| 971 | 13 | {1: 96, 3: 93} |
| 1028 | 15 | {2: 49, 1: 48, 3: 47, 0: 45} |

### 256x240_4096x3840x8192

`v9_tile/tmp/tile_n/att_r1/sweep_att/256x240_4096x3840x8192/ui_output_agent_22804_dispatch_60`: 12 waves, 63 iterations each. Per iteration and wave: 4267 cycles, 427 of them MFMA idle (loop eff 90.0%); MFMA stall 1416 pipe-full, 3 pipe-empty; unattributed 0.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| issue | 133 | 31.1% |
| memory: buffer_load issue stall | 87 | 20.3% |
| dependency hole | 80 | 18.7% |
| barrier | 47 | 11.1% |
| lds: ds_read stall | 30 | 7.0% |
| other stall | 27 | 6.4% |
| memory: vmcnt wait | 20 | 4.6% |
| mfma stall, pipe empty | 3 | 0.8% |
| lds: lgkmcnt wait | 0 | 0.0% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 910 | `buffer_load_dwordx4 v30, s[0:3], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 22.0 | 0.0 | 5.2% |
| 658 | `s_waitcnt vmcnt(20) lgkmcnt(0)` | `v_accvgpr_write_b32 a211, v0` | s_waitcnt vmcnt+lgkmcnt | 0.0 | 19.7 | 0.0 | 4.6% |
| 732 | `ds_read_b128 v[192:195], v0` | `v_add_u32_e32 v0, 0x149a0, v14` | ds_read | 8.0 | 2.0 | 4.0 | 3.3% |
| 798 | `ds_read_b128 v[112:115], v0` | `v_add_u32_e32 v0, 0x128c0, v14` | ds_read | 8.0 | 2.0 | 4.0 | 3.3% |
| 990 | `ds_read_b128 v[192:195], v0` | `v_add_u32_e32 v0, 0x16aa0, v14` | ds_read | 8.0 | 2.0 | 4.0 | 3.3% |
| 1052 | `ds_read_b128 v[120:123], v0` | `v_add_u32_e32 v0, 0x107c0, v14` | ds_read | 8.0 | 2.0 | 4.0 | 3.3% |
| 705 | `buffer_load_dwordx4 v17, s[0:3], 0 offen lds` | `v_mfma_f32_16x16x32_bf16 a[196:199], v[1` | buffer_load | 0.0 | 12.1 | 0.4 | 2.9% |
| 659 | `v_mfma_f32_16x16x32_bf16 a[236:239], v[120:123], v[92:95], a` | `s_waitcnt vmcnt(20) lgkmcnt(0)` | mfma | 11.8 | 0.0 | 0.0 | 2.8% |
| 729 | `s_barrier` | `v_mfma_f32_16x16x32_bf16 a[180:183], v[1` | s_barrier | 0.0 | 11.7 | 0.0 | 2.7% |
| 956 | `buffer_load_dwordx4 v32, s[0:3], 0 offen lds` | `v_mfma_f32_16x16x32_bf16 a[204:207], v[1` | buffer_load | 0.0 | 10.9 | 0.0 | 2.6% |
| 713 | `buffer_load_dwordx4 v18, s[0:3], 0 offen lds` | `v_mfma_f32_16x16x32_bf16 a[188:191], v[1` | buffer_load | 0.0 | 10.0 | 0.5 | 2.5% |
| 902 | `buffer_load_dwordx4 v29, s[0:3], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 8.9 | 0.0 | 2.1% |
| 697 | `buffer_load_dwordx4 v16, s[0:3], 0 offen lds` | `v_mfma_f32_16x16x32_bf16 a[204:207], v[1` | buffer_load | 0.0 | 8.6 | 0.3 | 2.1% |
| 741 | `ds_read_b128 v[184:187], v0` | `v_add_u32_e32 v0, 0x1cd60, v14` | ds_read | 4.0 | 0.0 | 4.0 | 1.9% |
| 746 | `ds_read_b128 v[176:179], v0` | `v_add_u32_e32 v0, 0x1ee40, v13` | ds_read | 4.0 | 0.0 | 4.0 | 1.9% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 659 | 660 `s_barrier` | 1 | 1 | 12 | 0 | yes | 18 | 17 |
| 661 | 723 `s_waitcnt lgkmcnt(11)` | 31 | 32 | 372 | 88 | yes | 45 | 4 |
| 728 | 729 `s_barrier` | 1 | 1 | 12 | 0 | yes | 12 | 23 |
| 730 | 794 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 29 | 36 | 348 | 100 | yes | 58 | 4 |
| 797 | 855 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 27 | 32 | 324 | 72 | yes | 29 | 4 |
| 856 | 856 `s_barrier` | 0 | 1 | 0 | 0 | yes | 6 | 14 |
| 857 | 919 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 28 | 35 | 328 | 83 | yes | 61 | 4 |
| 920 | 921 `s_barrier` | 1 | 1 | 12 | 0 | yes | 8 | 19 |
| 922 | 982 `s_waitcnt lgkmcnt(11)` | 31 | 30 | 372 | 80 | yes | 24 | 4 |
| 989 | 1048 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 29 | 31 | 348 | 80 | yes | 45 | 4 |
| 1051 | 1109 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 27 | 32 | 324 | 72 | yes | 29 | 4 |
| 1110 | 1110 `s_barrier` | 0 | 1 | 0 | 0 | yes | 6 | 14 |
| 1111 | 658 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 28 | 42 | 328 | 111 | yes | 77 | 20 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 658 | `s_waitcnt vmcnt(20) lgkmcnt(0)` | vmcnt | 0.98 | 20 | 2715 | 154.0 |
| 658 | `s_waitcnt vmcnt(20) lgkmcnt(0)` | lgkmcnt | 0.98 | 20 | 343 | 19.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 660 | 19 | {3: 189} |
| 729 | 30 | {3: 189} |
| 796 | 15 | {3: 189} |
| 856 | 14 | {3: 189} |
| 921 | 32 | {3: 189} |
| 988 | 14 | {3: 189} |
| 1050 | 14 | {3: 189} |
| 1110 | 14 | {3: 189} |

### 256x256_4096x4096x8192

`v9_tile/tmp/tile_n/att_r1/sweep_att/256x256_4096x4096x8192/ui_output_agent_62853_dispatch_58`: 12 waves, 63 iterations each. Per iteration and wave: 4283 cycles, 187 of them MFMA idle (loop eff 95.6%); MFMA stall 1652 pipe-full, 1 pipe-empty; unattributed 0.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| issue | 69 | 36.8% |
| dependency hole | 48 | 25.6% |
| lds: lgkmcnt wait | 20 | 10.6% |
| barrier | 19 | 10.1% |
| other stall | 16 | 8.6% |
| lds: ds_read stall | 8 | 4.3% |
| memory: buffer_load issue stall | 7 | 3.7% |
| mfma stall, pipe empty | 1 | 0.3% |
| memory: vmcnt wait | 0 | 0.0% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 634 | `s_waitcnt lgkmcnt(7)` | `v_accvgpr_write_b32 a195, v1` | s_waitcnt lgkmcnt | 0.0 | 19.7 | 0.0 | 10.6% |
| 774 | `ds_read_b128 v[64:67], v39` | `v_add_u32_e32 v39, 0x149c0, v1` | ds_read | 8.0 | 2.0 | 4.0 | 7.5% |
| 967 | `ds_read_b128 v[144:147], v39` | `v_add_u32_e32 v39, 0x1cda0, v1` | ds_read | 8.0 | 2.0 | 4.0 | 7.5% |
| 1030 | `ds_read_b128 v[96:99], v39` | `v_add_u32_e32 v39, 0x107c0, v1` | ds_read | 8.0 | 2.0 | 4.0 | 7.5% |
| 707 | `ds_read_b128 v[144:147], v39` | `v_add_u32_e32 v39, 0x18ba0, v1` | ds_read | 8.0 | 2.0 | 4.0 | 7.5% |
| 635 | `v_mfma_f32_16x16x32_bf16 a[252:255], v[96:99], v[88:91], a[2` | `s_waitcnt lgkmcnt(7)` | mfma | 11.8 | 0.0 | 0.0 | 6.3% |
| 1160 | `s_cbranch_scc1 64649` | `s_cmpk_lt_u32 s18, 0x7c` | other | 4.0 | 0.0 | 4.0 | 4.3% |
| 704 | `s_barrier` | `v_mfma_f32_16x16x32_bf16 a[184:187], v[7` | s_barrier | 0.0 | 4.9 | 0.0 | 2.6% |
| 641 | `ds_read_b128 v[128:131], v0 offset:33760` | `s_barrier` | ds_read | 0.0 | 0.0 | 4.0 | 2.1% |
| 706 | `v_add_u32_e32 v39, 0x18ba0, v1` | `v_add_u32_e32 v1, 0, v38` | other | 0.0 | 0.0 | 4.0 | 2.1% |
| 726 | `s_mov_b32 s7, s3` | `s_mov_b32 s6, s2` | other | 0.0 | 0.0 | 4.0 | 2.1% |
| 727 | `s_mov_b32 m0, s29` | `s_mov_b32 s7, s3` | other | 0.0 | 0.0 | 4.0 | 2.1% |
| 839 | `ds_read_b128 v[136:139], v0 offset:16896` | `s_barrier` | ds_read | 0.0 | 0.0 | 4.0 | 2.1% |
| 904 | `ds_read_b128 v[128:131], v0 offset:50656` | `s_barrier` | ds_read | 0.0 | 0.0 | 4.0 | 2.1% |
| 1095 | `ds_read_b128 v[88:91], v0` | `s_barrier` | ds_read | 0.0 | 0.0 | 4.0 | 2.1% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 635 | 636 `s_waitcnt lgkmcnt(6)` | 1 | 1 | 12 | 0 | yes | 12 | 4 |
| 641 | 698 `s_waitcnt lgkmcnt(7)` | 29 | 29 | 348 | 72 | yes | 14 | 4 |
| 705 | 766 `s_waitcnt lgkmcnt(7)` | 29 | 33 | 348 | 88 | yes | 30 | 4 |
| 773 | 836 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 31 | 33 | 372 | 68 | yes | 18 | 4 |
| 839 | 897 `s_waitcnt lgkmcnt(7)` | 29 | 30 | 348 | 64 | yes | 6 | 4 |
| 904 | 959 `s_waitcnt lgkmcnt(7)` | 29 | 27 | 348 | 64 | yes | 6 | 4 |
| 966 | 1022 `s_waitcnt lgkmcnt(7)` | 29 | 28 | 348 | 68 | yes | 18 | 4 |
| 1029 | 1092 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 31 | 33 | 372 | 68 | yes | 18 | 4 |
| 1095 | 634 `s_waitcnt lgkmcnt(7)` | 29 | 38 | 348 | 92 | yes | 46 | 20 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 634 | `s_waitcnt lgkmcnt(7)` | lgkmcnt | 0.98 | 20 | 490 | 29.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 640 | 9 | {3: 185, 2: 3, 1: 1} |
| 704 | 15 | {3: 163, 1: 23, 2: 3} |
| 772 | 9 | {3: 181, 1: 6, 2: 2} |
| 838 | 9 | {3: 189} |
| 903 | 9 | {3: 189} |
| 965 | 9 | {3: 189} |
| 1028 | 9 | {3: 185, 1: 4} |
| 1094 | 9 | {3: 189} |

