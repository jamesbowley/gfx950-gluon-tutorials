### 256x128_4096x2048x8192

`v9_tile/tmp/tile_n/att_r3/sweep_att/256x128_4096x2048x8192/ui_output_agent_12426_dispatch_70`: 12 waves, 63 iterations each. Per iteration and wave: 2684 cycles, 636 of them MFMA idle (loop eff 76.3%); MFMA stall 845 pipe-full, 5 pipe-empty; unattributed -4.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| issue | 212 | 33.3% |
| memory: buffer_load issue stall | 201 | 31.6% |
| barrier | 89 | 14.0% |
| lds: ds_read stall | 80 | 12.6% |
| other stall | 28 | 4.4% |
| memory: vmcnt wait | 20 | 3.1% |
| dependency hole | 6 | 0.9% |
| mfma stall, pipe empty | 5 | 0.8% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 773 | `buffer_load_dwordx4 v32, s[0:3], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 39.7 | 0.9 | 6.4% |
| 540 | `buffer_load_dwordx4 v7, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 33.0 | 0.9 | 5.3% |
| 532 | `buffer_load_dwordx4 v6, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 29.8 | 0.8 | 4.8% |
| 479 | `s_barrier` | `s_waitcnt vmcnt(14) lgkmcnt(0)` | s_barrier | 0.0 | 27.9 | 0.0 | 4.4% |
| 765 | `buffer_load_dwordx4 v31, s[0:3], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 18.2 | 1.7 | 3.1% |
| 478 | `s_waitcnt vmcnt(14) lgkmcnt(0)` | `v_accvgpr_write_b32 a99, v84` | s_waitcnt vmcnt+lgkmcnt | 0.0 | 19.7 | 0.0 | 3.1% |
| 548 | `buffer_load_dwordx4 v8, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 15.8 | 0.6 | 2.6% |
| 719 | `buffer_load_dwordx4 v28, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 14.9 | 1.0 | 2.5% |
| 727 | `buffer_load_dwordx4 v29, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 13.6 | 1.1 | 2.3% |
| 558 | `s_barrier` | `s_waitcnt vmcnt(16) lgkmcnt(0)` | s_barrier | 0.0 | 14.1 | 0.0 | 2.2% |
| 745 | `s_barrier` | `s_waitcnt vmcnt(16) lgkmcnt(0)` | s_barrier | 0.0 | 13.7 | 0.0 | 2.2% |
| 527 | `buffer_load_dwordx4 v5, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 8.2 | 4.0 | 1.9% |
| 516 | `s_barrier` | `s_waitcnt vmcnt(14) lgkmcnt(0)` | s_barrier | 0.0 | 11.1 | 0.0 | 1.7% |
| 482 | `ds_read_b128 v[108:111], v15 offset:34016` | `ds_read_b128 v[104:107], v15 offset:3382` | ds_read | 0.0 | 5.7 | 4.0 | 1.5% |
| 707 | `s_barrier` | `s_waitcnt vmcnt(16) lgkmcnt(0)` | s_barrier | 0.0 | 8.8 | 0.0 | 1.4% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 479 | 479 `s_barrier` | 0 | 1 | 0 | 0 | yes | 28 | 28 |
| 480 | 515 `s_waitcnt vmcnt(14) lgkmcnt(0)` | 16 | 20 | 184 | 57 | yes | 37 | 4 |
| 516 | 516 `s_barrier` | 0 | 1 | 0 | 0 | yes | 11 | 19 |
| 517 | 557 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 16 | 25 | 183 | 70 | yes | 149 | 4 |
| 558 | 558 `s_barrier` | 0 | 1 | 0 | 0 | yes | 14 | 22 |
| 559 | 595 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 16 | 21 | 183 | 54 | yes | 49 | 4 |
| 596 | 596 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 597 | 633 `s_waitcnt vmcnt(14) lgkmcnt(0)` | 16 | 21 | 180 | 55 | yes | 28 | 4 |
| 635 | 668 `s_waitcnt vmcnt(14) lgkmcnt(0)` | 16 | 18 | 184 | 49 | yes | 24 | 4 |
| 670 | 706 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 16 | 21 | 183 | 54 | yes | 53 | 4 |
| 707 | 707 `s_barrier` | 0 | 1 | 0 | 0 | yes | 9 | 17 |
| 708 | 744 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 16 | 21 | 183 | 54 | yes | 84 | 4 |
| 745 | 745 `s_barrier` | 0 | 1 | 0 | 0 | yes | 14 | 22 |
| 746 | 478 `s_waitcnt vmcnt(14) lgkmcnt(0)` | 16 | 28 | 180 | 83 | yes | 127 | 20 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 478 | `s_waitcnt vmcnt(14) lgkmcnt(0)` | vmcnt | 0.98 | 20 | 1712 | 86.0 |
| 478 | `s_waitcnt vmcnt(14) lgkmcnt(0)` | lgkmcnt | 0.98 | 20 | 266 | 11.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 479 | 40 | {1: 144, 3: 45} |
| 516 | 24 | {1: 188, 3: 1} |
| 558 | 31 | {1: 149, 3: 40} |
| 596 | 13 | {1: 130, 3: 59} |
| 634 | 11 | {1: 126, 3: 63} |
| 669 | 11 | {1: 126, 3: 63} |
| 707 | 18 | {1: 139, 3: 50} |
| 745 | 29 | {1: 153, 3: 36} |

### 256x144_4096x2304x8192

`v9_tile/tmp/tile_n/att_r3/sweep_att/256x144_4096x2304x8192/ui_output_agent_60489_dispatch_68`: 12 waves, 63 iterations each. Per iteration and wave: 3132 cycles, 828 of them MFMA idle (loop eff 73.6%); MFMA stall 922 pipe-full, 1 pipe-empty; unattributed -0.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| issue | 300 | 36.3% |
| lds: ds_read stall | 202 | 24.4% |
| memory: buffer_load issue stall | 135 | 16.3% |
| barrier | 78 | 9.5% |
| dependency hole | 63 | 7.6% |
| other stall | 29 | 3.5% |
| memory: vmcnt wait | 20 | 2.4% |
| mfma stall, pipe empty | 1 | 0.2% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 640 | `buffer_load_dwordx4 v19, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 36.3 | 0.1 | 4.4% |
| 532 | `s_barrier` | `s_waitcnt vmcnt(16) lgkmcnt(0)` | s_barrier | 0.0 | 24.1 | 0.0 | 2.9% |
| 632 | `buffer_load_dwordx4 v18, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 20.3 | 2.5 | 2.8% |
| 531 | `s_waitcnt vmcnt(16) lgkmcnt(0)` | `v_accvgpr_write_b32 a115, v0` | s_waitcnt vmcnt+lgkmcnt | 0.0 | 19.5 | 0.0 | 2.4% |
| 673 | `ds_read_b128 v[40:43], v16 offset:17664` | `ds_read_b128 v[60:63], v16 offset:17472` | ds_read | 0.0 | 12.0 | 4.0 | 1.9% |
| 850 | `ds_read_b128 v[56:59], v16 offset:768` | `ds_read_b128 v[64:67], v16 offset:576` | ds_read | 0.0 | 11.8 | 4.0 | 1.9% |
| 539 | `ds_read_b128 v[104:107], v16 offset:34528` | `ds_read_b128 v[132:135], v16 offset:3433` | ds_read | 0.0 | 11.8 | 4.0 | 1.9% |
| 538 | `ds_read_b128 v[132:135], v16 offset:34336` | `ds_read_b128 v[128:131], v16 offset:3427` | ds_read | 0.0 | 10.8 | 4.2 | 1.8% |
| 672 | `ds_read_b128 v[60:63], v16 offset:17472` | `ds_read_b128 v[56:59], v16 offset:17408` | ds_read | 0.0 | 11.0 | 4.0 | 1.8% |
| 722 | `ds_read_b128 v[132:135], v16 offset:51232` | `ds_read_b128 v[128:131], v16 offset:5116` | ds_read | 0.0 | 10.0 | 5.0 | 1.8% |
| 849 | `ds_read_b128 v[64:67], v16 offset:576` | `ds_read_b128 v[68:71], v16 offset:512` | ds_read | 0.0 | 10.8 | 4.2 | 1.8% |
| 723 | `ds_read_b128 v[104:107], v16 offset:51424` | `ds_read_b128 v[132:135], v16 offset:5123` | ds_read | 0.0 | 11.0 | 4.0 | 1.8% |
| 595 | `buffer_load_dwordx4 v7, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 14.1 | 0.6 | 1.8% |
| 875 | `buffer_load_dwordx4 v36, s[0:3], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 13.9 | 0.5 | 1.7% |
| 883 | `buffer_load_dwordx4 v37, s[0:3], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 13.8 | 0.5 | 1.7% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 532 | 532 `s_barrier` | 0 | 1 | 0 | 0 | yes | 24 | 24 |
| 533 | 572 `s_waitcnt vmcnt(15) lgkmcnt(0)` | 16 | 24 | 185 | 74 | yes | 84 | 4 |
| 573 | 573 `s_barrier` | 0 | 1 | 0 | 0 | yes | 14 | 22 |
| 574 | 620 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 16 | 31 | 185 | 92 | yes | 142 | 4 |
| 621 | 621 `s_barrier` | 0 | 1 | 0 | 0 | yes | 11 | 19 |
| 622 | 665 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 20 | 24 | 232 | 55 | yes | 98 | 4 |
| 666 | 666 `s_barrier` | 0 | 1 | 0 | 0 | yes | 6 | 14 |
| 667 | 715 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 20 | 29 | 232 | 74 | yes | 75 | 4 |
| 716 | 716 `s_barrier` | 0 | 1 | 0 | 0 | yes | 6 | 14 |
| 717 | 754 `s_waitcnt vmcnt(15) lgkmcnt(0)` | 16 | 22 | 184 | 71 | yes | 71 | 4 |
| 755 | 755 `s_barrier` | 0 | 1 | 0 | 0 | yes | 6 | 14 |
| 756 | 797 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 16 | 26 | 185 | 78 | yes | 89 | 4 |
| 798 | 798 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 799 | 842 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 20 | 24 | 232 | 56 | yes | 43 | 4 |
| 843 | 843 `s_barrier` | 0 | 1 | 0 | 0 | yes | 6 | 14 |
| 844 | 531 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 20 | 36 | 231 | 104 | yes | 148 | 19 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 531 | `s_waitcnt vmcnt(16) lgkmcnt(0)` | vmcnt | 0.98 | 20 | 2008 | 98.0 |
| 531 | `s_waitcnt vmcnt(16) lgkmcnt(0)` | lgkmcnt | 0.98 | 20 | 304 | 15.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 532 | 35 | {3: 120, 1: 67, 0: 1, 2: 1} |
| 573 | 28 | {3: 122, 1: 67} |
| 621 | 23 | {3: 121, 1: 65, 0: 2, 2: 1} |
| 666 | 14 | {3: 124, 1: 65} |
| 716 | 14 | {3: 126, 1: 63} |
| 755 | 14 | {3: 126, 1: 63} |
| 798 | 13 | {3: 125, 1: 64} |
| 843 | 14 | {3: 124, 1: 65} |

### 256x160_4096x2560x8192

`v9_tile/tmp/tile_n/att_r3/sweep_att/256x160_4096x2560x8192/ui_output_agent_44736_dispatch_66`: 12 waves, 63 iterations each. Per iteration and wave: 3136 cycles, 576 of them MFMA idle (loop eff 81.6%); MFMA stall 1017 pipe-full, 3 pipe-empty; unattributed -1.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| memory: buffer_load issue stall | 183 | 31.7% |
| issue | 171 | 29.8% |
| barrier | 90 | 15.6% |
| lds: ds_read stall | 61 | 10.6% |
| dependency hole | 25 | 4.3% |
| other stall | 24 | 4.2% |
| memory: vmcnt wait | 20 | 3.4% |
| mfma stall, pipe empty | 3 | 0.5% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 863 | `buffer_load_dwordx4 v35, s[0:3], 0 offen lds` | `v_mfma_f32_16x16x32_bf16 a[20:23], v[128` | buffer_load | 0.0 | 30.4 | 0.7 | 5.4% |
| 578 | `buffer_load_dwordx4 v6, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 24.5 | 0.7 | 4.4% |
| 523 | `s_barrier` | `s_waitcnt vmcnt(16) lgkmcnt(0)` | s_barrier | 0.0 | 23.4 | 0.0 | 4.1% |
| 560 | `s_barrier` | `s_waitcnt vmcnt(15) lgkmcnt(0)` | s_barrier | 0.0 | 20.1 | 0.0 | 3.5% |
| 586 | `buffer_load_dwordx4 v7, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 19.1 | 0.8 | 3.5% |
| 522 | `s_waitcnt vmcnt(16) lgkmcnt(0)` | `v_accvgpr_write_b32 a131, v39` | s_waitcnt vmcnt+lgkmcnt | 0.0 | 19.7 | 0.0 | 3.4% |
| 871 | `buffer_load_dwordx4 v36, s[0:3], 0 offen lds` | `v_mfma_f32_16x16x32_bf16 a[12:15], v[144` | buffer_load | 0.0 | 17.6 | 0.6 | 3.2% |
| 594 | `buffer_load_dwordx4 v8, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 15.1 | 0.5 | 2.7% |
| 566 | `ds_read_b128 v[124:127], v37 offset:64` | `ds_read_b128 v[120:123], v37` | ds_read | 0.0 | 10.0 | 4.0 | 2.4% |
| 784 | `ds_read_b128 v[80:83], v39` | `v_add_u32_e32 v39, 0x107c0, v16` | ds_read | 8.0 | 2.0 | 4.0 | 2.4% |
| 855 | `buffer_load_dwordx4 v34, s[0:3], 0 offen lds` | `v_mfma_f32_16x16x32_bf16 a[28:31], v[128` | buffer_load | 0.0 | 13.0 | 0.5 | 2.4% |
| 746 | `ds_read_b128 v[140:143], v39` | `v_add_u32_e32 v39, 0x19c00, v16` | ds_read | 7.0 | 2.0 | 4.0 | 2.3% |
| 741 | `ds_read_b128 v[124:127], v39` | `v_add_u32_e32 v39, 0x16aa0, v16` | ds_read | 6.0 | 2.0 | 5.0 | 2.3% |
| 573 | `buffer_load_dwordx4 v5, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 8.5 | 4.0 | 2.2% |
| 605 | `s_barrier` | `v_mfma_f32_16x16x32_bf16 a[96:99], v[132` | s_barrier | 0.0 | 12.0 | 0.0 | 2.1% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 523 | 523 `s_barrier` | 0 | 1 | 0 | 0 | yes | 23 | 23 |
| 524 | 559 `s_waitcnt vmcnt(15) lgkmcnt(0)` | 16 | 20 | 184 | 57 | yes | 43 | 4 |
| 560 | 560 `s_barrier` | 0 | 1 | 0 | 0 | yes | 20 | 28 |
| 561 | 603 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 16 | 27 | 183 | 80 | yes | 152 | 4 |
| 604 | 605 `s_barrier` | 1 | 1 | 12 | 0 | yes | 12 | 23 |
| 606 | 652 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 23 | 24 | 276 | 48 | yes | 13 | 4 |
| 655 | 703 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 23 | 26 | 276 | 56 | yes | 11 | 4 |
| 704 | 704 `s_barrier` | 0 | 1 | 0 | 0 | yes | 6 | 14 |
| 705 | 738 `s_waitcnt vmcnt(15) lgkmcnt(0)` | 16 | 18 | 184 | 49 | yes | 26 | 4 |
| 739 | 739 `s_barrier` | 0 | 1 | 0 | 0 | yes | 8 | 16 |
| 740 | 780 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 16 | 25 | 185 | 73 | yes | 88 | 4 |
| 781 | 782 `s_barrier` | 1 | 1 | 12 | 0 | yes | 8 | 19 |
| 783 | 830 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 23 | 25 | 276 | 52 | yes | 42 | 4 |
| 831 | 832 `s_barrier` | 1 | 1 | 12 | 0 | yes | 9 | 20 |
| 833 | 522 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 23 | 33 | 276 | 84 | yes | 111 | 20 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 522 | `s_waitcnt vmcnt(16) lgkmcnt(0)` | vmcnt | 0.98 | 20 | 2063 | 110.0 |
| 522 | `s_waitcnt vmcnt(16) lgkmcnt(0)` | lgkmcnt | 0.98 | 20 | 352 | 16.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 523 | 36 | {1: 104, 3: 85} |
| 560 | 39 | {1: 101, 3: 88} |
| 605 | 33 | {1: 106, 3: 83} |
| 654 | 14 | {1: 98, 3: 91} |
| 704 | 15 | {3: 96, 1: 93} |
| 739 | 18 | {1: 96, 3: 93} |
| 782 | 23 | {1: 101, 3: 87, 0: 1} |
| 832 | 26 | {1: 99, 3: 90} |

### 256x176_4096x2816x8192

`v9_tile/tmp/tile_n/att_r3/sweep_att/256x176_4096x2816x8192/ui_output_agent_27457_dispatch_64`: 12 waves, 63 iterations each. Per iteration and wave: 3423 cycles, 607 of them MFMA idle (loop eff 82.3%); MFMA stall 1098 pipe-full, 3 pipe-empty; unattributed 0.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| issue | 260 | 42.9% |
| lds: ds_read stall | 139 | 22.9% |
| dependency hole | 80 | 13.2% |
| barrier | 55 | 9.0% |
| other stall | 26 | 4.2% |
| memory: buffer_load issue stall | 25 | 4.1% |
| memory: vmcnt wait | 20 | 3.3% |
| mfma stall, pipe empty | 3 | 0.4% |
| lds: lgkmcnt wait | 0 | 0.0% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 572 | `s_waitcnt vmcnt(18) lgkmcnt(0)` | `v_accvgpr_write_b32 a147, v0` | s_waitcnt vmcnt+lgkmcnt | 0.0 | 19.7 | 0.0 | 3.3% |
| 580 | `ds_read_b128 v[108:111], v3 offset:34528` | `ds_read_b128 v[132:135], v3 offset:34336` | ds_read | 0.0 | 12.0 | 4.0 | 2.6% |
| 793 | `ds_read_b128 v[108:111], v3 offset:51424` | `ds_read_b128 v[132:135], v3 offset:51232` | ds_read | 0.0 | 12.0 | 4.0 | 2.6% |
| 792 | `ds_read_b128 v[132:135], v3 offset:51232` | `ds_read_b128 v[124:127], v3 offset:51168` | ds_read | 0.0 | 11.0 | 4.0 | 2.5% |
| 579 | `ds_read_b128 v[132:135], v3 offset:34336` | `ds_read_b128 v[124:127], v3 offset:34272` | ds_read | 0.0 | 11.0 | 4.0 | 2.5% |
| 573 | `s_barrier` | `s_waitcnt vmcnt(18) lgkmcnt(0)` | s_barrier | 0.0 | 14.7 | 0.0 | 2.4% |
| 672 | `ds_read_b128 v[92:95], v0` | `v_add_u32_e32 v0, 0x128c0, v21` | ds_read | 8.0 | 2.0 | 4.0 | 2.3% |
| 878 | `ds_read_b128 v[96:99], v0` | `v_add_u32_e32 v0, 0x107c0, v21` | ds_read | 8.0 | 2.0 | 4.0 | 2.3% |
| 622 | `ds_read_b128 v[164:167], v0` | `v_add_u32_e32 v0, 0x18b80, v21` | ds_read | 8.0 | 2.0 | 4.0 | 2.3% |
| 617 | `ds_read_b128 v[180:183], v0` | `v_add_u32_e32 v0, 0x149a0, v21` | ds_read | 7.3 | 2.0 | 4.0 | 2.2% |
| 832 | `ds_read_b128 v[176:179], v0` | `v_add_u32_e32 v0, 0x19c00, v21` | ds_read | 7.0 | 2.0 | 4.3 | 2.2% |
| 625 | `ds_read_b128 v[156:159], v0` | `v_add_u32_e32 v0, 0x1ac60, v20` | ds_read | 8.0 | 1.0 | 4.0 | 2.1% |
| 827 | `ds_read_b128 v[160:163], v0` | `v_add_u32_e32 v0, 0x16aa0, v21` | ds_read | 6.0 | 2.0 | 5.0 | 2.1% |
| 835 | `ds_read_b128 v[184:187], v0` | `v_add_u32_e32 v0, 0x1b460, v20` | ds_read | 7.7 | 1.0 | 4.0 | 2.1% |
| 627 | `ds_read_b128 v[160:163], v0` | `v_add_u32_e32 v0, 0x1ac60, v19` | ds_read | 8.0 | 0.0 | 4.0 | 2.0% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 573 | 573 `s_barrier` | 0 | 1 | 0 | 0 | yes | 15 | 15 |
| 574 | 613 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 16 | 24 | 184 | 73 | yes | 87 | 4 |
| 614 | 614 `s_barrier` | 0 | 1 | 0 | 0 | yes | 11 | 19 |
| 615 | 664 `s_waitcnt lgkmcnt(7)` | 16 | 34 | 185 | 104 | yes | 137 | 4 |
| 671 | 721 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 25 | 26 | 300 | 52 | yes | 21 | 4 |
| 722 | 722 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 723 | 785 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 28 | 35 | 328 | 83 | yes | 29 | 4 |
| 786 | 786 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 787 | 824 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 16 | 22 | 184 | 65 | yes | 72 | 4 |
| 825 | 825 `s_barrier` | 0 | 1 | 0 | 0 | yes | 6 | 14 |
| 826 | 870 `s_waitcnt lgkmcnt(7)` | 16 | 29 | 184 | 90 | yes | 113 | 4 |
| 877 | 927 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 25 | 26 | 300 | 52 | yes | 21 | 4 |
| 928 | 928 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 929 | 572 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 28 | 42 | 328 | 111 | yes | 71 | 20 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 572 | `s_waitcnt vmcnt(18) lgkmcnt(0)` | vmcnt | 0.98 | 20 | 2298 | 124.0 |
| 572 | `s_waitcnt vmcnt(18) lgkmcnt(0)` | lgkmcnt | 0.98 | 20 | 337 | 19.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 573 | 16 | {3: 184, 1: 4, 2: 1} |
| 614 | 26 | {3: 179, 1: 9, 2: 1} |
| 670 | 16 | {3: 184, 1: 5} |
| 722 | 13 | {3: 189} |
| 786 | 13 | {3: 189} |
| 825 | 15 | {3: 189} |
| 876 | 14 | {3: 188, 1: 1} |
| 928 | 14 | {3: 189} |

### 256x192_4096x3072x8192

`v9_tile/tmp/tile_n/att_r3/sweep_att/256x192_4096x3072x8192/ui_output_agent_20780_dispatch_64`: 12 waves, 63 iterations each. Per iteration and wave: 3495 cycles, 423 of them MFMA idle (loop eff 87.9%); MFMA stall 1303 pipe-full, 6 pipe-empty; unattributed 0.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| issue | 168 | 39.7% |
| lds: ds_read stall | 96 | 22.8% |
| barrier | 46 | 11.0% |
| dependency hole | 43 | 10.2% |
| other stall | 24 | 5.7% |
| memory: buffer_load issue stall | 20 | 4.7% |
| lds: lgkmcnt wait | 19 | 4.4% |
| mfma stall, pipe empty | 6 | 1.4% |
| memory: vmcnt wait | 0 | 0.0% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 562 | `s_waitcnt lgkmcnt(7)` | `v_accvgpr_write_b32 a163, v37` | s_waitcnt lgkmcnt | 0.0 | 18.8 | 0.0 | 4.4% |
| 613 | `buffer_load_dwordx4 v11, s[0:3], 0 offen lds` | `v_mfma_f32_16x16x32_bf16 a[140:143], v[6` | buffer_load | 0.0 | 14.2 | 0.5 | 3.5% |
| 693 | `ds_read_b128 v[98:101], v37` | `v_add_u32_e32 v37, 0x128c0, v10` | ds_read | 8.0 | 2.0 | 4.0 | 3.3% |
| 698 | `ds_read_b128 v[78:81], v37` | `v_add_u32_e32 v37, 0x1ac80, v10` | ds_read | 8.0 | 2.0 | 4.0 | 3.3% |
| 845 | `ds_read_b128 v[138:141], v37` | `v_add_u32_e32 v37, 0x16aa0, v10` | ds_read | 8.0 | 2.0 | 4.0 | 3.3% |
| 898 | `ds_read_b128 v[98:101], v37` | `v_add_u32_e32 v37, 0x107c0, v10` | ds_read | 8.0 | 2.0 | 4.0 | 3.3% |
| 903 | `ds_read_b128 v[78:81], v37` | `v_add_u32_e32 v37, 0x18b80, v10` | ds_read | 8.0 | 2.0 | 4.0 | 3.3% |
| 635 | `s_barrier` | `v_mfma_f32_16x16x32_bf16 a[128:131], v[9` | s_barrier | 0.0 | 13.7 | 0.0 | 3.2% |
| 700 | `ds_read_b128 v[62:65], v37 offset:4224` | `ds_read_b128 v[82:85], v37 offset:64` | ds_read | 0.0 | 9.0 | 4.0 | 3.1% |
| 905 | `ds_read_b128 v[62:65], v37 offset:4224` | `ds_read_b128 v[82:85], v37 offset:64` | ds_read | 0.0 | 9.0 | 4.0 | 3.1% |
| 701 | `ds_read_b128 v[66:69], v37 offset:4288` | `ds_read_b128 v[62:65], v37 offset:4224` | ds_read | 0.0 | 8.0 | 4.0 | 2.8% |
| 906 | `ds_read_b128 v[66:69], v37 offset:4288` | `ds_read_b128 v[62:65], v37 offset:4224` | ds_read | 0.0 | 8.0 | 4.0 | 2.8% |
| 694 | `ds_read_b128 v[94:97], v37 offset:64` | `ds_read_b128 v[98:101], v37` | ds_read | 0.0 | 4.0 | 4.0 | 1.9% |
| 695 | `ds_read_b128 v[86:89], v37 offset:4224` | `ds_read_b128 v[94:97], v37 offset:64` | ds_read | 0.0 | 4.0 | 4.0 | 1.9% |
| 696 | `ds_read_b128 v[90:93], v37 offset:4288` | `ds_read_b128 v[86:89], v37 offset:4224` | ds_read | 0.0 | 4.0 | 4.0 | 1.9% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 567 | 568 `s_barrier` | 1 | 1 | 12 | 0 | yes | 5 | 15 |
| 569 | 626 `s_waitcnt lgkmcnt(7)` | 29 | 29 | 348 | 72 | yes | 23 | 4 |
| 629 | 633 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 3 | 2 | 36 | 0 | yes | 0 | 4 |
| 634 | 635 `s_barrier` | 1 | 1 | 12 | 0 | yes | 14 | 24 |
| 636 | 690 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 27 | 28 | 324 | 64 | yes | 15 | 4 |
| 692 | 734 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 16 | 27 | 185 | 75 | yes | 107 | 4 |
| 735 | 735 `s_barrier` | 0 | 1 | 0 | 0 | yes | 8 | 16 |
| 736 | 772 `s_waitcnt lgkmcnt(7)` | 16 | 21 | 184 | 49 | yes | 28 | 4 |
| 779 | 834 `s_waitcnt lgkmcnt(7)` | 29 | 27 | 348 | 64 | yes | 6 | 4 |
| 837 | 841 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 3 | 2 | 36 | 0 | yes | 0 | 4 |
| 844 | 895 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 27 | 25 | 324 | 52 | yes | 18 | 4 |
| 897 | 939 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 16 | 27 | 185 | 75 | yes | 107 | 4 |
| 940 | 940 `s_barrier` | 0 | 1 | 0 | 0 | yes | 8 | 16 |
| 941 | 562 `s_waitcnt lgkmcnt(7)` | 16 | 28 | 184 | 79 | yes | 72 | 19 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 562 | `s_waitcnt lgkmcnt(7)` | lgkmcnt | 0.98 | 19 | 307 | 16.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 568 | 19 | {3: 110, 1: 79} |
| 635 | 35 | {3: 98, 1: 91} |
| 691 | 12 | {3: 125, 1: 64} |
| 735 | 23 | {3: 97, 1: 92} |
| 778 | 12 | {3: 126, 1: 63} |
| 843 | 12 | {3: 126, 1: 63} |
| 896 | 12 | {3: 126, 1: 63} |
| 940 | 23 | {1: 98, 3: 91} |

### 256x208_4096x3328x8192

`v9_tile/tmp/tile_n/att_r3/sweep_att/256x208_4096x3328x8192/ui_output_agent_62483_dispatch_62`: 12 waves, 63 iterations each. Per iteration and wave: 3898 cycles, 570 of them MFMA idle (loop eff 85.4%); MFMA stall 1359 pipe-full, 1 pipe-empty; unattributed -0.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| issue | 205 | 36.0% |
| lds: ds_read stall | 150 | 26.3% |
| dependency hole | 80 | 14.0% |
| barrier | 65 | 11.4% |
| other stall | 27 | 4.7% |
| memory: buffer_load issue stall | 22 | 3.9% |
| memory: vmcnt wait | 20 | 3.5% |
| mfma stall, pipe empty | 1 | 0.2% |
| lds: lgkmcnt wait | 0 | 0.0% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 616 | `s_waitcnt vmcnt(18) lgkmcnt(0)` | `v_accvgpr_write_b32 a179, v0` | s_waitcnt vmcnt+lgkmcnt | 0.0 | 19.7 | 0.0 | 3.5% |
| 809 | `ds_read_b128 v[44:47], v1 offset:17664` | `ds_read_b128 v[64:67], v1 offset:17472` | ds_read | 0.0 | 12.0 | 4.0 | 2.8% |
| 1037 | `ds_read_b128 v[60:63], v1 offset:768` | `ds_read_b128 v[72:75], v1 offset:576` | ds_read | 0.0 | 12.0 | 4.0 | 2.8% |
| 808 | `ds_read_b128 v[64:67], v1 offset:17472` | `ds_read_b128 v[60:63], v1 offset:17408` | ds_read | 0.0 | 11.0 | 4.0 | 2.6% |
| 1036 | `ds_read_b128 v[72:75], v1 offset:576` | `ds_read_b128 v[68:71], v1 offset:512` | ds_read | 0.0 | 11.0 | 4.0 | 2.6% |
| 754 | `ds_read_b128 v[116:119], v0` | `v_add_u32_e32 v0, 0x128c0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 2.5% |
| 759 | `ds_read_b128 v[88:91], v0` | `v_add_u32_e32 v0, 0x1ac80, v12` | ds_read | 8.0 | 2.0 | 4.0 | 2.5% |
| 924 | `ds_read_b128 v[176:179], v0` | `v_add_u32_e32 v0, 0x16aa0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 2.5% |
| 982 | `ds_read_b128 v[116:119], v0` | `v_add_u32_e32 v0, 0x107c0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 2.5% |
| 987 | `ds_read_b128 v[100:103], v0` | `v_add_u32_e32 v0, 0x18b80, v12` | ds_read | 8.0 | 2.0 | 4.0 | 2.5% |
| 1030 | `s_barrier` | `s_waitcnt vmcnt(19) lgkmcnt(0)` | s_barrier | 0.0 | 14.0 | 0.0 | 2.5% |
| 692 | `ds_read_b128 v[176:179], v0` | `v_add_u32_e32 v0, 0x149a0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 2.5% |
| 802 | `s_barrier` | `s_waitcnt vmcnt(19) lgkmcnt(0)` | s_barrier | 0.0 | 14.0 | 0.0 | 2.5% |
| 989 | `ds_read_b128 v[84:87], v0 offset:4224` | `ds_read_b128 v[104:107], v0 offset:64` | ds_read | 0.0 | 9.0 | 4.0 | 2.3% |
| 761 | `ds_read_b128 v[84:87], v0 offset:4224` | `ds_read_b128 v[100:103], v0 offset:64` | ds_read | 0.0 | 9.0 | 4.0 | 2.3% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 617 | 618 `s_barrier` | 1 | 1 | 12 | 0 | yes | 17 | 16 |
| 619 | 681 `s_waitcnt lgkmcnt(11)` | 31 | 32 | 372 | 88 | yes | 26 | 4 |
| 684 | 688 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 3 | 2 | 36 | 0 | yes | 0 | 4 |
| 689 | 689 `s_barrier` | 0 | 1 | 0 | 0 | yes | 11 | 19 |
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
| 1031 | 616 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 20 | 36 | 231 | 102 | yes | 116 | 20 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 616 | `s_waitcnt vmcnt(18) lgkmcnt(0)` | vmcnt | 0.98 | 20 | 2431 | 130.0 |
| 616 | `s_waitcnt vmcnt(18) lgkmcnt(0)` | lgkmcnt | 0.98 | 20 | 271 | 15.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 618 | 18 | {3: 189} |
| 689 | 26 | {3: 188, 2: 1} |
| 752 | 15 | {3: 188, 1: 1} |
| 802 | 31 | {3: 189} |
| 853 | 15 | {3: 189} |
| 922 | 15 | {3: 189} |
| 980 | 15 | {3: 189} |
| 1030 | 31 | {3: 189} |

### 256x224_4096x3584x8192

`v9_tile/tmp/tile_n/att_r3/sweep_att/256x224_4096x3584x8192/ui_output_agent_29007_dispatch_60`: 12 waves, 63 iterations each. Per iteration and wave: 3920 cycles, 336 of them MFMA idle (loop eff 91.4%); MFMA stall 1452 pipe-full, 1 pipe-empty; unattributed -0.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| issue | 119 | 35.4% |
| dependency hole | 55 | 16.5% |
| barrier | 50 | 14.9% |
| other stall | 34 | 10.1% |
| memory: buffer_load issue stall | 29 | 8.7% |
| lds: ds_read stall | 28 | 8.2% |
| lds: lgkmcnt wait | 20 | 5.9% |
| mfma stall, pipe empty | 1 | 0.3% |
| memory: vmcnt wait | 0 | 0.0% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 604 | `s_waitcnt lgkmcnt(7)` | `v_accvgpr_write_b32 a195, v12` | s_waitcnt lgkmcnt | 0.0 | 19.7 | 0.0 | 5.9% |
| 679 | `ds_read_b128 v[146:149], v41` | `v_add_u32_e32 v41, 0x149a0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 4.2% |
| 740 | `ds_read_b128 v[94:97], v41` | `v_add_u32_e32 v41, 0x128c0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 4.2% |
| 916 | `ds_read_b128 v[150:153], v41` | `v_add_u32_e32 v41, 0x16aa0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 4.2% |
| 973 | `ds_read_b128 v[98:101], v41` | `v_add_u32_e32 v41, 0x107c0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 4.2% |
| 1028 | `s_barrier` | `v_mfma_f32_16x16x32_bf16 a[60:63], v[150` | s_barrier | 0.0 | 10.3 | 0.0 | 3.1% |
| 981 | `ds_read_b128 v[82:85], v41` | `v_add_u32_e32 v41, 0x18b80, v12` | ds_read | 6.0 | 0.0 | 4.0 | 3.0% |
| 748 | `ds_read_b128 v[74:77], v41` | `v_add_u32_e32 v41, 0x1ac80, v12` | ds_read | 5.5 | 0.1 | 4.0 | 2.9% |
| 676 | `s_barrier` | `s_waitcnt vmcnt(19) lgkmcnt(0)` | s_barrier | 0.0 | 9.5 | 0.0 | 2.8% |
| 610 | `s_barrier` | `v_mfma_f32_16x16x32_bf16 a[216:219], v[9` | s_barrier | 0.0 | 8.1 | 0.0 | 2.4% |
| 688 | `ds_read_b128 v[138:141], v41` | `v_add_u32_e32 v41, 0x1cd60, v12` | ds_read | 4.0 | 0.0 | 4.0 | 2.4% |
| 741 | `ds_read_b128 v[98:101], v41 offset:64` | `ds_read_b128 v[94:97], v41` | ds_read | 0.0 | 4.0 | 4.0 | 2.4% |
| 742 | `ds_read_b128 v[90:93], v41 offset:4224` | `ds_read_b128 v[98:101], v41 offset:64` | ds_read | 0.0 | 4.0 | 4.0 | 2.4% |
| 925 | `ds_read_b128 v[138:141], v41` | `v_add_u32_e32 v41, 0x1dde0, v12` | ds_read | 4.0 | 0.0 | 4.0 | 2.4% |
| 974 | `ds_read_b128 v[102:105], v41 offset:64` | `ds_read_b128 v[98:101], v41` | ds_read | 0.0 | 4.0 | 4.0 | 2.4% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 609 | 610 `s_barrier` | 1 | 1 | 12 | 0 | yes | 8 | 19 |
| 611 | 668 `s_waitcnt lgkmcnt(7)` | 29 | 29 | 348 | 72 | yes | 20 | 4 |
| 671 | 675 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 3 | 2 | 36 | 0 | yes | 0 | 4 |
| 676 | 676 `s_barrier` | 0 | 1 | 0 | 0 | yes | 10 | 18 |
| 677 | 737 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 28 | 33 | 328 | 87 | yes | 49 | 4 |
| 738 | 738 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 739 | 793 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 24 | 31 | 280 | 75 | yes | 55 | 4 |
| 796 | 844 `s_waitcnt lgkmcnt(7)` | 23 | 26 | 276 | 56 | yes | 7 | 4 |
| 851 | 906 `s_waitcnt lgkmcnt(7)` | 29 | 27 | 348 | 64 | yes | 7 | 4 |
| 909 | 913 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 3 | 2 | 36 | 0 | yes | 0 | 4 |
| 915 | 970 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 28 | 28 | 328 | 67 | yes | 29 | 4 |
| 971 | 971 `s_barrier` | 0 | 1 | 0 | 0 | yes | 6 | 14 |
| 972 | 1026 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 24 | 31 | 280 | 75 | yes | 60 | 4 |
| 1027 | 1028 `s_barrier` | 1 | 1 | 12 | 0 | yes | 10 | 21 |
| 1029 | 604 `s_waitcnt lgkmcnt(7)` | 23 | 33 | 276 | 84 | yes | 57 | 20 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 604 | `s_waitcnt lgkmcnt(7)` | lgkmcnt | 0.98 | 20 | 405 | 23.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 610 | 22 | {3: 139, 2: 27, 1: 23} |
| 676 | 21 | {3: 172, 1: 17} |
| 738 | 13 | {3: 186, 1: 3} |
| 795 | 15 | {3: 61, 1: 48, 0: 48, 2: 32} |
| 850 | 12 | {3: 142, 2: 31, 1: 16} |
| 914 | 13 | {3: 189} |
| 971 | 14 | {3: 188, 1: 1} |
| 1028 | 26 | {2: 48, 1: 48, 3: 47, 0: 46} |

### 256x240_4096x3840x8192

`v9_tile/tmp/tile_n/att_r3/sweep_att/256x240_4096x3840x8192/ui_output_agent_48617_dispatch_60`: 12 waves, 63 iterations each. Per iteration and wave: 4153 cycles, 313 of them MFMA idle (loop eff 92.5%); MFMA stall 1416 pipe-full, 3 pipe-empty; unattributed 0.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| issue | 125 | 39.9% |
| dependency hole | 80 | 25.5% |
| lds: ds_read stall | 29 | 9.2% |
| barrier | 25 | 8.0% |
| other stall | 23 | 7.5% |
| memory: vmcnt wait | 20 | 6.3% |
| memory: buffer_load issue stall | 8 | 2.7% |
| mfma stall, pipe empty | 3 | 1.0% |
| lds: lgkmcnt wait | 0 | 0.0% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 658 | `s_waitcnt vmcnt(20) lgkmcnt(0)` | `v_accvgpr_write_b32 a211, v0` | s_waitcnt vmcnt+lgkmcnt | 0.0 | 19.7 | 0.0 | 6.3% |
| 798 | `ds_read_b128 v[112:115], v0` | `v_add_u32_e32 v0, 0x128c0, v14` | ds_read | 8.0 | 2.0 | 4.0 | 4.5% |
| 990 | `ds_read_b128 v[192:195], v0` | `v_add_u32_e32 v0, 0x16aa0, v14` | ds_read | 8.0 | 2.0 | 4.0 | 4.5% |
| 1052 | `ds_read_b128 v[120:123], v0` | `v_add_u32_e32 v0, 0x107c0, v14` | ds_read | 8.0 | 2.0 | 4.0 | 4.5% |
| 732 | `ds_read_b128 v[192:195], v0` | `v_add_u32_e32 v0, 0x149a0, v14` | ds_read | 8.0 | 2.0 | 4.0 | 4.5% |
| 659 | `v_mfma_f32_16x16x32_bf16 a[236:239], v[120:123], v[92:95], a` | `s_waitcnt vmcnt(20) lgkmcnt(0)` | mfma | 11.8 | 0.0 | 0.0 | 3.8% |
| 741 | `ds_read_b128 v[184:187], v0` | `v_add_u32_e32 v0, 0x1cd60, v14` | ds_read | 4.0 | 0.0 | 4.0 | 2.6% |
| 746 | `ds_read_b128 v[176:179], v0` | `v_add_u32_e32 v0, 0x1ee40, v13` | ds_read | 4.0 | 0.0 | 4.0 | 2.6% |
| 749 | `ds_read_b128 v[180:183], v0` | `v_add_u32_e32 v0, 0x1ee40, v12` | ds_read | 4.0 | 0.0 | 4.0 | 2.6% |
| 809 | `ds_read_b128 v[96:99], v0` | `v_add_u32_e32 v0, 0x1ac80, v14` | ds_read | 4.0 | 0.0 | 4.0 | 2.6% |
| 858 | `ds_read_b128 v[52:55], v1 offset:16960` | `ds_read_b128 v[56:59], v1 offset:16896` | ds_read | 0.0 | 4.0 | 4.0 | 2.6% |
| 859 | `ds_read_b128 v[48:51], v1 offset:17152` | `ds_read_b128 v[52:55], v1 offset:16960` | ds_read | 0.0 | 4.0 | 4.0 | 2.6% |
| 999 | `ds_read_b128 v[180:183], v0` | `v_add_u32_e32 v0, 0x1dde0, v14` | ds_read | 4.0 | 0.0 | 4.0 | 2.6% |
| 1004 | `ds_read_b128 v[176:179], v0` | `v_add_u32_e32 v0, 0x1f640, v13` | ds_read | 4.0 | 0.0 | 4.0 | 2.6% |
| 1007 | `ds_read_b128 v[184:187], v0` | `v_add_u32_e32 v0, 0x1f640, v12` | ds_read | 4.0 | 0.0 | 4.0 | 2.6% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 659 | 660 `s_barrier` | 1 | 1 | 12 | 0 | yes | 15 | 13 |
| 661 | 723 `s_waitcnt lgkmcnt(11)` | 31 | 32 | 372 | 88 | yes | 13 | 4 |
| 730 | 794 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 29 | 36 | 348 | 100 | yes | 56 | 4 |
| 797 | 855 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 27 | 32 | 324 | 72 | yes | 28 | 4 |
| 857 | 919 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 28 | 35 | 328 | 83 | yes | 29 | 4 |
| 922 | 982 `s_waitcnt lgkmcnt(11)` | 31 | 30 | 372 | 80 | yes | 8 | 4 |
| 989 | 1048 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 29 | 31 | 348 | 80 | yes | 44 | 4 |
| 1051 | 1109 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 27 | 32 | 324 | 72 | yes | 28 | 4 |
| 1111 | 658 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 28 | 42 | 328 | 111 | yes | 69 | 20 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 658 | `s_waitcnt vmcnt(20) lgkmcnt(0)` | vmcnt | 0.98 | 20 | 2647 | 154.0 |
| 658 | `s_waitcnt vmcnt(20) lgkmcnt(0)` | lgkmcnt | 0.98 | 20 | 334 | 19.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 660 | 14 | {3: 125, 1: 63, 2: 1} |
| 729 | 16 | {3: 118, 1: 69, 2: 2} |
| 796 | 13 | {3: 102, 1: 87} |
| 856 | 13 | {3: 126, 1: 63} |
| 921 | 13 | {3: 126, 1: 63} |
| 988 | 13 | {3: 126, 1: 63} |
| 1050 | 13 | {3: 105, 1: 84} |
| 1110 | 13 | {3: 126, 1: 63} |

### 256x256_4096x4096x8192

`v9_tile/tmp/tile_n/att_r3/sweep_att/256x256_4096x4096x8192/ui_output_agent_33290_dispatch_58`: 12 waves, 63 iterations each. Per iteration and wave: 4277 cycles, 181 of them MFMA idle (loop eff 95.8%); MFMA stall 1652 pipe-full, 1 pipe-empty; unattributed 0.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| issue | 69 | 38.2% |
| dependency hole | 48 | 26.5% |
| lds: lgkmcnt wait | 20 | 10.9% |
| barrier | 17 | 9.4% |
| other stall | 16 | 9.0% |
| lds: ds_read stall | 8 | 4.4% |
| memory: buffer_load issue stall | 2 | 1.2% |
| mfma stall, pipe empty | 1 | 0.3% |
| memory: vmcnt wait | 0 | 0.0% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 634 | `s_waitcnt lgkmcnt(7)` | `v_accvgpr_write_b32 a195, v1` | s_waitcnt lgkmcnt | 0.0 | 19.7 | 0.0 | 10.9% |
| 774 | `ds_read_b128 v[64:67], v39` | `v_add_u32_e32 v39, 0x149c0, v1` | ds_read | 8.0 | 2.0 | 4.0 | 7.7% |
| 967 | `ds_read_b128 v[144:147], v39` | `v_add_u32_e32 v39, 0x1cda0, v1` | ds_read | 8.0 | 2.0 | 4.0 | 7.7% |
| 1030 | `ds_read_b128 v[96:99], v39` | `v_add_u32_e32 v39, 0x107c0, v1` | ds_read | 8.0 | 2.0 | 4.0 | 7.7% |
| 707 | `ds_read_b128 v[144:147], v39` | `v_add_u32_e32 v39, 0x18ba0, v1` | ds_read | 8.0 | 2.0 | 4.0 | 7.7% |
| 635 | `v_mfma_f32_16x16x32_bf16 a[252:255], v[96:99], v[88:91], a[2` | `s_waitcnt lgkmcnt(7)` | mfma | 11.8 | 0.0 | 0.0 | 6.5% |
| 1160 | `s_cbranch_scc1 64649` | `s_cmpk_lt_u32 s18, 0x7c` | other | 4.0 | 0.0 | 4.0 | 4.4% |
| 641 | `ds_read_b128 v[128:131], v0 offset:33760` | `s_barrier` | ds_read | 0.0 | 0.0 | 4.0 | 2.2% |
| 706 | `v_add_u32_e32 v39, 0x18ba0, v1` | `v_add_u32_e32 v1, 0, v38` | other | 0.0 | 0.0 | 4.0 | 2.2% |
| 726 | `s_mov_b32 s7, s3` | `s_mov_b32 s6, s2` | other | 0.0 | 0.0 | 4.0 | 2.2% |
| 727 | `s_mov_b32 m0, s29` | `s_mov_b32 s7, s3` | other | 0.0 | 0.0 | 4.0 | 2.2% |
| 839 | `ds_read_b128 v[136:139], v0 offset:16896` | `s_barrier` | ds_read | 0.0 | 0.0 | 4.0 | 2.2% |
| 904 | `ds_read_b128 v[128:131], v0 offset:50656` | `s_barrier` | ds_read | 0.0 | 0.0 | 4.0 | 2.2% |
| 1095 | `ds_read_b128 v[88:91], v0` | `s_barrier` | ds_read | 0.0 | 0.0 | 4.0 | 2.2% |
| 1157 | `s_addc_u32 s9, s9, 0` | `s_add_u32 s8, s8, 0x100` | other | 0.0 | 0.0 | 4.0 | 2.2% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 635 | 636 `s_waitcnt lgkmcnt(6)` | 1 | 1 | 12 | 0 | yes | 12 | 4 |
| 641 | 698 `s_waitcnt lgkmcnt(7)` | 29 | 29 | 348 | 72 | yes | 9 | 4 |
| 705 | 766 `s_waitcnt lgkmcnt(7)` | 29 | 33 | 348 | 88 | yes | 31 | 4 |
| 773 | 836 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 31 | 33 | 372 | 68 | yes | 18 | 4 |
| 839 | 897 `s_waitcnt lgkmcnt(7)` | 29 | 30 | 348 | 64 | yes | 6 | 4 |
| 904 | 959 `s_waitcnt lgkmcnt(7)` | 29 | 27 | 348 | 64 | yes | 6 | 4 |
| 966 | 1022 `s_waitcnt lgkmcnt(7)` | 29 | 28 | 348 | 68 | yes | 19 | 4 |
| 1029 | 1092 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 31 | 33 | 372 | 68 | yes | 18 | 4 |
| 1095 | 634 `s_waitcnt lgkmcnt(7)` | 29 | 38 | 348 | 92 | yes | 46 | 20 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 634 | `s_waitcnt lgkmcnt(7)` | lgkmcnt | 0.98 | 20 | 490 | 29.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 640 | 12 | {1: 186, 2: 3} |
| 704 | 14 | {1: 186, 2: 3} |
| 772 | 12 | {1: 189} |
| 838 | 12 | {1: 189} |
| 903 | 12 | {1: 189} |
| 965 | 12 | {1: 189} |
| 1028 | 12 | {1: 189} |
| 1094 | 12 | {1: 189} |

