### 256x128_4096x2048x8192

`v9_tile/tmp/tile_n/att_r2/sweep_att/256x128_4096x2048x8192/ui_output_agent_45373_dispatch_70`: 12 waves, 63 iterations each. Per iteration and wave: 2908 cycles, 860 of them MFMA idle (loop eff 70.4%); MFMA stall 845 pipe-full, 7 pipe-empty; unattributed -4.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| memory: buffer_load issue stall | 372 | 43.3% |
| issue | 219 | 25.5% |
| barrier | 133 | 15.5% |
| lds: ds_read stall | 80 | 9.3% |
| other stall | 28 | 3.3% |
| memory: vmcnt wait | 20 | 2.3% |
| mfma stall, pipe empty | 7 | 0.8% |
| dependency hole | 6 | 0.7% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 532 | `buffer_load_dwordx4 v6, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 45.6 | 1.3 | 5.4% |
| 727 | `buffer_load_dwordx4 v29, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 42.3 | 1.4 | 5.1% |
| 773 | `buffer_load_dwordx4 v32, s[0:3], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 38.3 | 1.0 | 4.6% |
| 719 | `buffer_load_dwordx4 v28, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 35.1 | 1.7 | 4.3% |
| 540 | `buffer_load_dwordx4 v7, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 32.4 | 1.0 | 3.9% |
| 479 | `s_barrier` | `s_waitcnt vmcnt(14) lgkmcnt(0)` | s_barrier | 0.0 | 30.2 | 0.0 | 3.5% |
| 707 | `s_barrier` | `s_waitcnt vmcnt(16) lgkmcnt(0)` | s_barrier | 0.0 | 24.7 | 0.0 | 2.9% |
| 745 | `s_barrier` | `s_waitcnt vmcnt(16) lgkmcnt(0)` | s_barrier | 0.0 | 21.6 | 0.0 | 2.5% |
| 478 | `s_waitcnt vmcnt(14) lgkmcnt(0)` | `v_accvgpr_write_b32 a99, v84` | s_waitcnt vmcnt+lgkmcnt | 0.0 | 19.7 | 0.0 | 2.3% |
| 548 | `buffer_load_dwordx4 v8, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 18.5 | 0.7 | 2.2% |
| 735 | `buffer_load_dwordx4 v30, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 17.6 | 1.1 | 2.2% |
| 697 | `buffer_load_dwordx4 v26, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 17.3 | 1.2 | 2.2% |
| 558 | `s_barrier` | `s_waitcnt vmcnt(16) lgkmcnt(0)` | s_barrier | 0.0 | 18.4 | 0.0 | 2.1% |
| 681 | `buffer_load_dwordx4 v24, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 17.3 | 1.0 | 2.1% |
| 765 | `buffer_load_dwordx4 v31, s[0:3], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 15.8 | 1.7 | 2.0% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 479 | 479 `s_barrier` | 0 | 1 | 0 | 0 | yes | 30 | 30 |
| 480 | 515 `s_waitcnt vmcnt(14) lgkmcnt(0)` | 16 | 20 | 185 | 57 | yes | 40 | 4 |
| 516 | 516 `s_barrier` | 0 | 1 | 0 | 0 | yes | 12 | 20 |
| 517 | 557 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 16 | 25 | 183 | 70 | yes | 173 | 4 |
| 558 | 558 `s_barrier` | 0 | 1 | 0 | 0 | yes | 18 | 26 |
| 559 | 595 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 16 | 21 | 183 | 54 | yes | 77 | 4 |
| 596 | 596 `s_barrier` | 0 | 1 | 0 | 0 | yes | 12 | 20 |
| 597 | 633 `s_waitcnt vmcnt(14) lgkmcnt(0)` | 16 | 21 | 179 | 55 | yes | 37 | 4 |
| 634 | 634 `s_barrier` | 0 | 1 | 0 | 0 | yes | 9 | 17 |
| 635 | 668 `s_waitcnt vmcnt(14) lgkmcnt(0)` | 16 | 18 | 185 | 49 | yes | 26 | 4 |
| 669 | 669 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 670 | 706 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 16 | 21 | 183 | 54 | yes | 100 | 4 |
| 707 | 707 `s_barrier` | 0 | 1 | 0 | 0 | yes | 25 | 33 |
| 708 | 744 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 16 | 21 | 183 | 54 | yes | 155 | 4 |
| 745 | 745 `s_barrier` | 0 | 1 | 0 | 0 | yes | 22 | 30 |
| 746 | 478 `s_waitcnt vmcnt(14) lgkmcnt(0)` | 16 | 28 | 179 | 83 | yes | 123 | 20 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 478 | `s_waitcnt vmcnt(14) lgkmcnt(0)` | vmcnt | 0.98 | 20 | 1874 | 86.0 |
| 478 | `s_waitcnt vmcnt(14) lgkmcnt(0)` | lgkmcnt | 0.98 | 20 | 262 | 11.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 479 | 43 | {3: 127, 1: 62} |
| 516 | 24 | {1: 179, 3: 10} |
| 558 | 39 | {3: 105, 1: 84} |
| 596 | 24 | {3: 126, 1: 63} |
| 634 | 19 | {3: 159, 1: 30} |
| 669 | 12 | {3: 181, 1: 8} |
| 707 | 46 | {1: 98, 3: 91} |
| 745 | 42 | {1: 103, 3: 86} |

### 256x144_4096x2304x8192

`v9_tile/tmp/tile_n/att_r2/sweep_att/256x144_4096x2304x8192/ui_output_agent_58889_dispatch_68`: 12 waves, 63 iterations each. Per iteration and wave: 3084 cycles, 780 of them MFMA idle (loop eff 74.7%); MFMA stall 923 pipe-full, 1 pipe-empty; unattributed -0.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| issue | 297 | 38.1% |
| lds: ds_read stall | 204 | 26.1% |
| memory: buffer_load issue stall | 90 | 11.5% |
| barrier | 76 | 9.8% |
| dependency hole | 63 | 8.1% |
| other stall | 29 | 3.7% |
| memory: vmcnt wait | 20 | 2.5% |
| mfma stall, pipe empty | 1 | 0.2% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 532 | `s_barrier` | `s_waitcnt vmcnt(16) lgkmcnt(0)` | s_barrier | 0.0 | 24.8 | 0.0 | 3.2% |
| 875 | `buffer_load_dwordx4 v36, s[0:3], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 23.7 | 0.6 | 3.1% |
| 531 | `s_waitcnt vmcnt(16) lgkmcnt(0)` | `v_accvgpr_write_b32 a115, v0` | s_waitcnt vmcnt+lgkmcnt | 0.0 | 19.7 | 0.0 | 2.5% |
| 673 | `ds_read_b128 v[40:43], v16 offset:17664` | `ds_read_b128 v[60:63], v16 offset:17472` | ds_read | 0.0 | 12.0 | 4.0 | 2.0% |
| 850 | `ds_read_b128 v[56:59], v16 offset:768` | `ds_read_b128 v[64:67], v16 offset:576` | ds_read | 0.0 | 12.0 | 4.0 | 2.0% |
| 539 | `ds_read_b128 v[104:107], v16 offset:34528` | `ds_read_b128 v[132:135], v16 offset:3433` | ds_read | 0.0 | 12.0 | 4.0 | 2.0% |
| 883 | `buffer_load_dwordx4 v37, s[0:3], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 14.5 | 0.5 | 1.9% |
| 538 | `ds_read_b128 v[132:135], v16 offset:34336` | `ds_read_b128 v[128:131], v16 offset:3427` | ds_read | 0.0 | 11.0 | 4.0 | 1.9% |
| 672 | `ds_read_b128 v[60:63], v16 offset:17472` | `ds_read_b128 v[56:59], v16 offset:17408` | ds_read | 0.0 | 11.0 | 4.0 | 1.9% |
| 722 | `ds_read_b128 v[132:135], v16 offset:51232` | `ds_read_b128 v[128:131], v16 offset:5116` | ds_read | 0.0 | 10.0 | 5.0 | 1.9% |
| 849 | `ds_read_b128 v[64:67], v16 offset:576` | `ds_read_b128 v[68:71], v16 offset:512` | ds_read | 0.0 | 11.0 | 4.0 | 1.9% |
| 723 | `ds_read_b128 v[104:107], v16 offset:51424` | `ds_read_b128 v[132:135], v16 offset:5123` | ds_read | 0.0 | 11.0 | 4.0 | 1.9% |
| 800 | `ds_read_b128 v[92:95], v0` | `v_add_u32_e32 v0, 0x107c0, v39` | ds_read | 8.0 | 2.0 | 4.0 | 1.8% |
| 623 | `ds_read_b128 v[76:79], v0` | `v_add_u32_e32 v0, 0x128c0, v39` | ds_read | 8.0 | 2.0 | 4.0 | 1.8% |
| 595 | `buffer_load_dwordx4 v7, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 12.8 | 0.6 | 1.7% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 532 | 532 `s_barrier` | 0 | 1 | 0 | 0 | yes | 25 | 25 |
| 533 | 572 `s_waitcnt vmcnt(15) lgkmcnt(0)` | 16 | 24 | 185 | 73 | yes | 83 | 4 |
| 573 | 573 `s_barrier` | 0 | 1 | 0 | 0 | yes | 13 | 21 |
| 574 | 620 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 16 | 31 | 185 | 92 | yes | 141 | 4 |
| 621 | 621 `s_barrier` | 0 | 1 | 0 | 0 | yes | 12 | 20 |
| 622 | 665 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 20 | 24 | 232 | 55 | yes | 42 | 4 |
| 666 | 666 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 667 | 715 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 20 | 29 | 232 | 74 | yes | 75 | 4 |
| 716 | 716 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 717 | 754 `s_waitcnt vmcnt(15) lgkmcnt(0)` | 16 | 22 | 184 | 71 | yes | 71 | 4 |
| 755 | 755 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 756 | 797 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 16 | 26 | 185 | 78 | yes | 89 | 4 |
| 798 | 798 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 799 | 842 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 20 | 24 | 232 | 55 | yes | 43 | 4 |
| 843 | 843 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 844 | 531 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 20 | 36 | 232 | 102 | yes | 161 | 20 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 531 | `s_waitcnt vmcnt(16) lgkmcnt(0)` | vmcnt | 0.98 | 20 | 2018 | 98.0 |
| 531 | `s_waitcnt vmcnt(16) lgkmcnt(0)` | lgkmcnt | 0.98 | 20 | 316 | 15.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 532 | 35 | {1: 102, 3: 86, 0: 1} |
| 573 | 27 | {1: 102, 3: 87} |
| 621 | 26 | {1: 98, 3: 86, 2: 3, 0: 2} |
| 666 | 13 | {1: 101, 3: 88} |
| 716 | 15 | {1: 101, 3: 88} |
| 755 | 15 | {1: 101, 3: 88} |
| 798 | 13 | {1: 101, 3: 88} |
| 843 | 14 | {1: 101, 3: 88} |

### 256x160_4096x2560x8192

`v9_tile/tmp/tile_n/att_r2/sweep_att/256x160_4096x2560x8192/ui_output_agent_26783_dispatch_66`: 12 waves, 63 iterations each. Per iteration and wave: 3094 cycles, 534 of them MFMA idle (loop eff 82.7%); MFMA stall 1016 pipe-full, 2 pipe-empty; unattributed -2.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| issue | 176 | 32.9% |
| memory: buffer_load issue stall | 142 | 26.6% |
| barrier | 81 | 15.2% |
| lds: ds_read stall | 63 | 11.9% |
| other stall | 26 | 4.9% |
| dependency hole | 25 | 4.7% |
| memory: vmcnt wait | 20 | 3.7% |
| mfma stall, pipe empty | 2 | 0.4% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 578 | `buffer_load_dwordx4 v6, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 30.5 | 0.8 | 5.9% |
| 863 | `buffer_load_dwordx4 v35, s[0:3], 0 offen lds` | `v_mfma_f32_16x16x32_bf16 a[20:23], v[128` | buffer_load | 0.0 | 24.5 | 0.6 | 4.7% |
| 523 | `s_barrier` | `s_waitcnt vmcnt(16) lgkmcnt(0)` | s_barrier | 0.0 | 23.4 | 0.0 | 4.4% |
| 586 | `buffer_load_dwordx4 v7, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 21.7 | 0.8 | 4.2% |
| 522 | `s_waitcnt vmcnt(16) lgkmcnt(0)` | `v_accvgpr_write_b32 a131, v39` | s_waitcnt vmcnt+lgkmcnt | 0.0 | 19.7 | 0.0 | 3.7% |
| 560 | `s_barrier` | `s_waitcnt vmcnt(15) lgkmcnt(0)` | s_barrier | 0.0 | 16.9 | 0.0 | 3.2% |
| 871 | `buffer_load_dwordx4 v36, s[0:3], 0 offen lds` | `v_mfma_f32_16x16x32_bf16 a[12:15], v[144` | buffer_load | 0.0 | 14.0 | 0.6 | 2.7% |
| 566 | `ds_read_b128 v[124:127], v37 offset:64` | `ds_read_b128 v[120:123], v37` | ds_read | 0.0 | 10.0 | 4.0 | 2.6% |
| 784 | `ds_read_b128 v[80:83], v39` | `v_add_u32_e32 v39, 0x107c0, v16` | ds_read | 8.0 | 2.0 | 4.0 | 2.6% |
| 573 | `buffer_load_dwordx4 v5, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 9.8 | 4.0 | 2.6% |
| 746 | `ds_read_b128 v[140:143], v39` | `v_add_u32_e32 v39, 0x19c00, v16` | ds_read | 7.0 | 2.0 | 4.7 | 2.6% |
| 741 | `ds_read_b128 v[124:127], v39` | `v_add_u32_e32 v39, 0x16aa0, v16` | ds_read | 6.0 | 2.0 | 5.0 | 2.4% |
| 605 | `s_barrier` | `v_mfma_f32_16x16x32_bf16 a[96:99], v[132` | s_barrier | 0.0 | 9.6 | 0.0 | 1.8% |
| 565 | `ds_read_b128 v[120:123], v37` | `ds_read_b128 v[144:147], v0 offset:4288` | ds_read | 0.0 | 4.0 | 5.0 | 1.7% |
| 594 | `buffer_load_dwordx4 v8, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 8.4 | 0.4 | 1.6% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 523 | 523 `s_barrier` | 0 | 1 | 0 | 0 | yes | 23 | 23 |
| 524 | 559 `s_waitcnt vmcnt(15) lgkmcnt(0)` | 16 | 20 | 185 | 57 | yes | 41 | 4 |
| 560 | 560 `s_barrier` | 0 | 1 | 0 | 0 | yes | 17 | 25 |
| 561 | 603 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 16 | 27 | 183 | 80 | yes | 156 | 4 |
| 604 | 605 `s_barrier` | 1 | 1 | 12 | 0 | yes | 10 | 21 |
| 606 | 652 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 23 | 24 | 275 | 48 | yes | 10 | 4 |
| 655 | 703 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 23 | 26 | 276 | 56 | yes | 9 | 4 |
| 704 | 704 `s_barrier` | 0 | 1 | 0 | 0 | yes | 6 | 14 |
| 705 | 738 `s_waitcnt vmcnt(15) lgkmcnt(0)` | 16 | 18 | 185 | 49 | yes | 27 | 4 |
| 739 | 739 `s_barrier` | 0 | 1 | 0 | 0 | yes | 7 | 15 |
| 740 | 780 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 16 | 25 | 184 | 74 | yes | 84 | 4 |
| 781 | 782 `s_barrier` | 1 | 1 | 12 | 0 | yes | 7 | 17 |
| 783 | 830 `s_waitcnt vmcnt(17) lgkmcnt(0)` | 23 | 25 | 276 | 52 | yes | 32 | 4 |
| 831 | 832 `s_barrier` | 1 | 1 | 12 | 0 | yes | 8 | 19 |
| 833 | 522 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 23 | 33 | 276 | 84 | yes | 95 | 20 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 522 | `s_waitcnt vmcnt(16) lgkmcnt(0)` | vmcnt | 0.98 | 20 | 2027 | 110.0 |
| 522 | `s_waitcnt vmcnt(16) lgkmcnt(0)` | lgkmcnt | 0.98 | 20 | 335 | 16.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 523 | 32 | {3: 185, 1: 2, 0: 2} |
| 560 | 31 | {3: 187, 1: 2} |
| 605 | 26 | {3: 187, 1: 2} |
| 654 | 14 | {3: 187, 1: 2} |
| 704 | 14 | {3: 187, 1: 2} |
| 739 | 15 | {3: 187, 1: 2} |
| 782 | 20 | {3: 187, 1: 2} |
| 832 | 24 | {3: 187, 1: 1, 0: 1} |

### 256x176_4096x2816x8192

`v9_tile/tmp/tile_n/att_r2/sweep_att/256x176_4096x2816x8192/ui_output_agent_10625_dispatch_64`: 12 waves, 63 iterations each. Per iteration and wave: 3411 cycles, 595 of them MFMA idle (loop eff 82.6%); MFMA stall 1098 pipe-full, 3 pipe-empty; unattributed 0.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| issue | 259 | 43.5% |
| lds: ds_read stall | 138 | 23.3% |
| dependency hole | 80 | 13.5% |
| barrier | 51 | 8.5% |
| other stall | 25 | 4.2% |
| memory: vmcnt wait | 20 | 3.3% |
| memory: buffer_load issue stall | 19 | 3.2% |
| mfma stall, pipe empty | 3 | 0.5% |
| lds: lgkmcnt wait | 0 | 0.0% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 572 | `s_waitcnt vmcnt(18) lgkmcnt(0)` | `v_accvgpr_write_b32 a147, v0` | s_waitcnt vmcnt+lgkmcnt | 0.0 | 19.7 | 0.0 | 3.3% |
| 580 | `ds_read_b128 v[108:111], v3 offset:34528` | `ds_read_b128 v[132:135], v3 offset:34336` | ds_read | 0.0 | 12.0 | 4.0 | 2.7% |
| 793 | `ds_read_b128 v[108:111], v3 offset:51424` | `ds_read_b128 v[132:135], v3 offset:51232` | ds_read | 0.0 | 12.0 | 4.0 | 2.7% |
| 579 | `ds_read_b128 v[132:135], v3 offset:34336` | `ds_read_b128 v[124:127], v3 offset:34272` | ds_read | 0.0 | 11.0 | 4.0 | 2.5% |
| 792 | `ds_read_b128 v[132:135], v3 offset:51232` | `ds_read_b128 v[124:127], v3 offset:51168` | ds_read | 0.0 | 11.0 | 4.0 | 2.5% |
| 573 | `s_barrier` | `s_waitcnt vmcnt(18) lgkmcnt(0)` | s_barrier | 0.0 | 14.1 | 0.0 | 2.4% |
| 672 | `ds_read_b128 v[92:95], v0` | `v_add_u32_e32 v0, 0x128c0, v21` | ds_read | 8.0 | 2.0 | 4.0 | 2.4% |
| 878 | `ds_read_b128 v[96:99], v0` | `v_add_u32_e32 v0, 0x107c0, v21` | ds_read | 8.0 | 2.0 | 4.0 | 2.4% |
| 622 | `ds_read_b128 v[164:167], v0` | `v_add_u32_e32 v0, 0x18b80, v21` | ds_read | 8.0 | 2.0 | 4.0 | 2.4% |
| 617 | `ds_read_b128 v[180:183], v0` | `v_add_u32_e32 v0, 0x149a0, v21` | ds_read | 7.5 | 2.0 | 4.0 | 2.3% |
| 832 | `ds_read_b128 v[176:179], v0` | `v_add_u32_e32 v0, 0x19c00, v21` | ds_read | 7.0 | 2.0 | 4.3 | 2.2% |
| 625 | `ds_read_b128 v[156:159], v0` | `v_add_u32_e32 v0, 0x1ac60, v20` | ds_read | 8.0 | 1.0 | 4.0 | 2.2% |
| 827 | `ds_read_b128 v[160:163], v0` | `v_add_u32_e32 v0, 0x16aa0, v21` | ds_read | 6.0 | 2.0 | 5.0 | 2.2% |
| 835 | `ds_read_b128 v[184:187], v0` | `v_add_u32_e32 v0, 0x1b460, v20` | ds_read | 7.7 | 1.0 | 4.0 | 2.1% |
| 627 | `ds_read_b128 v[160:163], v0` | `v_add_u32_e32 v0, 0x1ac60, v19` | ds_read | 8.0 | 0.0 | 4.0 | 2.0% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 573 | 573 `s_barrier` | 0 | 1 | 0 | 0 | yes | 14 | 14 |
| 574 | 613 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 16 | 24 | 184 | 73 | yes | 88 | 4 |
| 614 | 614 `s_barrier` | 0 | 1 | 0 | 0 | yes | 10 | 18 |
| 615 | 664 `s_waitcnt lgkmcnt(7)` | 16 | 34 | 185 | 104 | yes | 137 | 4 |
| 671 | 721 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 25 | 26 | 300 | 52 | yes | 19 | 4 |
| 723 | 785 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 28 | 35 | 328 | 83 | yes | 29 | 4 |
| 787 | 824 `s_waitcnt vmcnt(16) lgkmcnt(0)` | 16 | 22 | 184 | 65 | yes | 71 | 4 |
| 825 | 825 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 826 | 870 `s_waitcnt lgkmcnt(7)` | 16 | 29 | 184 | 90 | yes | 111 | 4 |
| 877 | 927 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 25 | 26 | 300 | 52 | yes | 19 | 4 |
| 929 | 572 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 28 | 42 | 328 | 111 | yes | 69 | 20 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 572 | `s_waitcnt vmcnt(18) lgkmcnt(0)` | vmcnt | 0.98 | 20 | 2294 | 124.0 |
| 572 | `s_waitcnt vmcnt(18) lgkmcnt(0)` | lgkmcnt | 0.98 | 20 | 335 | 19.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 573 | 15 | {3: 181, 1: 8} |
| 614 | 23 | {3: 175, 1: 14} |
| 670 | 16 | {3: 172, 1: 17} |
| 722 | 12 | {3: 189} |
| 786 | 12 | {3: 189} |
| 825 | 14 | {3: 189} |
| 876 | 13 | {3: 189} |
| 928 | 12 | {3: 189} |

### 256x192_4096x3072x8192

`v9_tile/tmp/tile_n/att_r2/sweep_att/256x192_4096x3072x8192/ui_output_agent_36703_dispatch_64`: 12 waves, 63 iterations each. Per iteration and wave: 3544 cycles, 472 of them MFMA idle (loop eff 86.7%); MFMA stall 1303 pipe-full, 6 pipe-empty; unattributed 0.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| issue | 169 | 35.7% |
| lds: ds_read stall | 96 | 20.4% |
| barrier | 63 | 13.3% |
| memory: buffer_load issue stall | 51 | 10.9% |
| dependency hole | 43 | 9.1% |
| other stall | 25 | 5.2% |
| lds: lgkmcnt wait | 19 | 4.0% |
| mfma stall, pipe empty | 6 | 1.3% |
| memory: vmcnt wait | 0 | 0.0% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 940 | `s_barrier` | `s_waitcnt vmcnt(18) lgkmcnt(0)` | s_barrier | 0.0 | 20.1 | 0.0 | 4.3% |
| 562 | `s_waitcnt lgkmcnt(7)` | `v_accvgpr_write_b32 a163, v37` | s_waitcnt lgkmcnt | 0.0 | 18.8 | 0.0 | 4.0% |
| 613 | `buffer_load_dwordx4 v11, s[0:3], 0 offen lds` | `v_mfma_f32_16x16x32_bf16 a[140:143], v[6` | buffer_load | 0.0 | 15.9 | 0.5 | 3.5% |
| 635 | `s_barrier` | `v_mfma_f32_16x16x32_bf16 a[128:131], v[9` | s_barrier | 0.0 | 15.0 | 0.0 | 3.2% |
| 693 | `ds_read_b128 v[98:101], v37` | `v_add_u32_e32 v37, 0x128c0, v10` | ds_read | 8.0 | 2.0 | 4.0 | 3.0% |
| 698 | `ds_read_b128 v[78:81], v37` | `v_add_u32_e32 v37, 0x1ac80, v10` | ds_read | 8.0 | 2.0 | 4.0 | 3.0% |
| 845 | `ds_read_b128 v[138:141], v37` | `v_add_u32_e32 v37, 0x16aa0, v10` | ds_read | 8.0 | 2.0 | 4.0 | 3.0% |
| 898 | `ds_read_b128 v[98:101], v37` | `v_add_u32_e32 v37, 0x107c0, v10` | ds_read | 8.0 | 2.0 | 4.0 | 3.0% |
| 903 | `ds_read_b128 v[78:81], v37` | `v_add_u32_e32 v37, 0x18b80, v10` | ds_read | 8.0 | 2.0 | 4.0 | 3.0% |
| 700 | `ds_read_b128 v[62:65], v37 offset:4224` | `ds_read_b128 v[82:85], v37 offset:64` | ds_read | 0.0 | 9.0 | 4.0 | 2.8% |
| 905 | `ds_read_b128 v[62:65], v37 offset:4224` | `ds_read_b128 v[82:85], v37 offset:64` | ds_read | 0.0 | 9.0 | 4.0 | 2.8% |
| 930 | `buffer_load_dwordx4 v33, s[4:7], 0 offen lds` | `s_nop 0` | buffer_load | 0.0 | 12.2 | 0.3 | 2.6% |
| 701 | `ds_read_b128 v[66:69], v37 offset:4288` | `ds_read_b128 v[62:65], v37 offset:4224` | ds_read | 0.0 | 8.0 | 4.0 | 2.5% |
| 906 | `ds_read_b128 v[66:69], v37 offset:4288` | `ds_read_b128 v[62:65], v37 offset:4224` | ds_read | 0.0 | 8.0 | 4.0 | 2.5% |
| 694 | `ds_read_b128 v[94:97], v37 offset:64` | `ds_read_b128 v[98:101], v37` | ds_read | 0.0 | 4.0 | 4.0 | 1.7% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 567 | 568 `s_barrier` | 1 | 1 | 12 | 0 | yes | 5 | 15 |
| 569 | 626 `s_waitcnt lgkmcnt(7)` | 29 | 29 | 348 | 72 | yes | 27 | 4 |
| 629 | 633 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 3 | 2 | 36 | 0 | yes | 0 | 4 |
| 634 | 635 `s_barrier` | 1 | 1 | 12 | 0 | yes | 15 | 25 |
| 636 | 690 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 27 | 28 | 324 | 64 | yes | 25 | 4 |
| 691 | 691 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 692 | 734 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 16 | 27 | 185 | 75 | yes | 108 | 4 |
| 735 | 735 `s_barrier` | 0 | 1 | 0 | 0 | yes | 8 | 16 |
| 736 | 772 `s_waitcnt lgkmcnt(7)` | 16 | 21 | 185 | 49 | yes | 28 | 4 |
| 779 | 834 `s_waitcnt lgkmcnt(7)` | 29 | 27 | 347 | 64 | yes | 6 | 4 |
| 837 | 841 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 3 | 2 | 36 | 0 | yes | 0 | 4 |
| 844 | 895 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 27 | 25 | 324 | 52 | yes | 19 | 4 |
| 896 | 896 `s_barrier` | 0 | 1 | 0 | 0 | yes | 6 | 14 |
| 897 | 939 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 16 | 27 | 185 | 75 | yes | 122 | 4 |
| 940 | 940 `s_barrier` | 0 | 1 | 0 | 0 | yes | 20 | 28 |
| 941 | 562 `s_waitcnt lgkmcnt(7)` | 16 | 28 | 185 | 79 | yes | 74 | 19 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 562 | `s_waitcnt lgkmcnt(7)` | lgkmcnt | 0.98 | 19 | 309 | 16.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 568 | 15 | {3: 168, 1: 21} |
| 635 | 34 | {3: 148, 1: 41} |
| 691 | 12 | {3: 174, 1: 15} |
| 735 | 20 | {3: 143, 1: 46} |
| 778 | 9 | {3: 189} |
| 843 | 9 | {3: 189} |
| 896 | 11 | {3: 181, 1: 8} |
| 940 | 39 | {3: 126, 1: 63} |

### 256x208_4096x3328x8192

`v9_tile/tmp/tile_n/att_r2/sweep_att/256x208_4096x3328x8192/ui_output_agent_4147_dispatch_62`: 12 waves, 63 iterations each. Per iteration and wave: 3911 cycles, 583 of them MFMA idle (loop eff 85.1%); MFMA stall 1359 pipe-full, 1 pipe-empty; unattributed 0.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| issue | 205 | 35.2% |
| lds: ds_read stall | 150 | 25.7% |
| dependency hole | 80 | 13.7% |
| barrier | 74 | 12.7% |
| memory: buffer_load issue stall | 27 | 4.7% |
| other stall | 26 | 4.5% |
| memory: vmcnt wait | 20 | 3.4% |
| mfma stall, pipe empty | 1 | 0.2% |
| lds: lgkmcnt wait | 0 | 0.0% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 616 | `s_waitcnt vmcnt(18) lgkmcnt(0)` | `v_accvgpr_write_b32 a179, v0` | s_waitcnt vmcnt+lgkmcnt | 0.0 | 19.7 | 0.0 | 3.4% |
| 1030 | `s_barrier` | `s_waitcnt vmcnt(19) lgkmcnt(0)` | s_barrier | 0.0 | 19.0 | 0.0 | 3.3% |
| 809 | `ds_read_b128 v[44:47], v1 offset:17664` | `ds_read_b128 v[64:67], v1 offset:17472` | ds_read | 0.0 | 12.0 | 4.0 | 2.7% |
| 1037 | `ds_read_b128 v[60:63], v1 offset:768` | `ds_read_b128 v[72:75], v1 offset:576` | ds_read | 0.0 | 12.0 | 4.0 | 2.7% |
| 689 | `s_barrier` | `s_waitcnt vmcnt(19) lgkmcnt(0)` | s_barrier | 0.0 | 15.5 | 0.0 | 2.7% |
| 808 | `ds_read_b128 v[64:67], v1 offset:17472` | `ds_read_b128 v[60:63], v1 offset:17408` | ds_read | 0.0 | 11.0 | 4.0 | 2.6% |
| 1036 | `ds_read_b128 v[72:75], v1 offset:576` | `ds_read_b128 v[68:71], v1 offset:512` | ds_read | 0.0 | 11.0 | 4.0 | 2.6% |
| 692 | `ds_read_b128 v[176:179], v0` | `v_add_u32_e32 v0, 0x149a0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 2.4% |
| 754 | `ds_read_b128 v[116:119], v0` | `v_add_u32_e32 v0, 0x128c0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 2.4% |
| 759 | `ds_read_b128 v[88:91], v0` | `v_add_u32_e32 v0, 0x1ac80, v12` | ds_read | 8.0 | 2.0 | 4.0 | 2.4% |
| 802 | `s_barrier` | `s_waitcnt vmcnt(19) lgkmcnt(0)` | s_barrier | 0.0 | 14.0 | 0.0 | 2.4% |
| 924 | `ds_read_b128 v[176:179], v0` | `v_add_u32_e32 v0, 0x16aa0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 2.4% |
| 982 | `ds_read_b128 v[116:119], v0` | `v_add_u32_e32 v0, 0x107c0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 2.4% |
| 987 | `ds_read_b128 v[100:103], v0` | `v_add_u32_e32 v0, 0x18b80, v12` | ds_read | 8.0 | 2.0 | 4.0 | 2.4% |
| 761 | `ds_read_b128 v[84:87], v0 offset:4224` | `ds_read_b128 v[100:103], v0 offset:64` | ds_read | 0.0 | 9.0 | 4.0 | 2.2% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 617 | 618 `s_barrier` | 1 | 1 | 12 | 0 | yes | 16 | 15 |
| 619 | 681 `s_waitcnt lgkmcnt(11)` | 31 | 32 | 372 | 88 | yes | 22 | 4 |
| 684 | 688 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 3 | 2 | 36 | 0 | yes | 0 | 4 |
| 689 | 689 `s_barrier` | 0 | 1 | 0 | 0 | yes | 15 | 23 |
| 690 | 751 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 28 | 34 | 328 | 91 | yes | 59 | 4 |
| 752 | 752 `s_barrier` | 0 | 1 | 0 | 0 | yes | 6 | 14 |
| 753 | 801 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 20 | 29 | 232 | 75 | yes | 86 | 4 |
| 802 | 802 `s_barrier` | 0 | 1 | 0 | 0 | yes | 14 | 22 |
| 803 | 851 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 20 | 29 | 232 | 74 | yes | 74 | 4 |
| 854 | 914 `s_waitcnt lgkmcnt(11)` | 31 | 30 | 372 | 80 | yes | 11 | 4 |
| 917 | 921 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 3 | 2 | 36 | 0 | yes | 0 | 4 |
| 922 | 922 `s_barrier` | 0 | 1 | 0 | 0 | yes | 6 | 14 |
| 923 | 979 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 28 | 29 | 328 | 71 | yes | 38 | 4 |
| 980 | 980 `s_barrier` | 0 | 1 | 0 | 0 | yes | 6 | 14 |
| 981 | 1029 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 20 | 29 | 232 | 75 | yes | 94 | 4 |
| 1030 | 1030 `s_barrier` | 0 | 1 | 0 | 0 | yes | 19 | 27 |
| 1031 | 616 `s_waitcnt vmcnt(18) lgkmcnt(0)` | 20 | 36 | 232 | 102 | yes | 114 | 20 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 616 | `s_waitcnt vmcnt(18) lgkmcnt(0)` | vmcnt | 0.98 | 20 | 2437 | 130.0 |
| 616 | `s_waitcnt vmcnt(18) lgkmcnt(0)` | lgkmcnt | 0.98 | 20 | 268 | 15.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 618 | 15 | {3: 189} |
| 689 | 29 | {3: 189} |
| 752 | 14 | {3: 189} |
| 802 | 29 | {3: 189} |
| 853 | 13 | {3: 189} |
| 922 | 14 | {3: 189} |
| 980 | 13 | {3: 189} |
| 1030 | 37 | {3: 189} |

### 256x224_4096x3584x8192

`v9_tile/tmp/tile_n/att_r2/sweep_att/256x224_4096x3584x8192/ui_output_agent_11490_dispatch_60`: 12 waves, 63 iterations each. Per iteration and wave: 3901 cycles, 317 of them MFMA idle (loop eff 91.9%); MFMA stall 1452 pipe-full, 1 pipe-empty; unattributed 0.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| issue | 119 | 37.4% |
| dependency hole | 55 | 17.5% |
| barrier | 40 | 12.6% |
| other stall | 34 | 10.6% |
| lds: ds_read stall | 28 | 8.7% |
| memory: buffer_load issue stall | 21 | 6.6% |
| lds: lgkmcnt wait | 20 | 6.2% |
| mfma stall, pipe empty | 1 | 0.4% |
| memory: vmcnt wait | 0 | 0.0% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 604 | `s_waitcnt lgkmcnt(7)` | `v_accvgpr_write_b32 a195, v12` | s_waitcnt lgkmcnt | 0.0 | 19.7 | 0.0 | 6.2% |
| 679 | `ds_read_b128 v[146:149], v41` | `v_add_u32_e32 v41, 0x149a0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 4.4% |
| 740 | `ds_read_b128 v[94:97], v41` | `v_add_u32_e32 v41, 0x128c0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 4.4% |
| 916 | `ds_read_b128 v[150:153], v41` | `v_add_u32_e32 v41, 0x16aa0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 4.4% |
| 973 | `ds_read_b128 v[98:101], v41` | `v_add_u32_e32 v41, 0x107c0, v12` | ds_read | 8.0 | 2.0 | 4.0 | 4.4% |
| 676 | `s_barrier` | `s_waitcnt vmcnt(19) lgkmcnt(0)` | s_barrier | 0.0 | 11.7 | 0.0 | 3.7% |
| 655 | `buffer_load_dwordx4 v13, s[0:3], 0 offen lds` | `v_mfma_f32_16x16x32_bf16 a[172:175], v[7` | buffer_load | 0.0 | 10.4 | 0.4 | 3.4% |
| 981 | `ds_read_b128 v[82:85], v41` | `v_add_u32_e32 v41, 0x18b80, v12` | ds_read | 6.0 | 0.0 | 4.0 | 3.2% |
| 748 | `ds_read_b128 v[74:77], v41` | `v_add_u32_e32 v41, 0x1ac80, v12` | ds_read | 5.4 | 0.2 | 4.0 | 3.0% |
| 688 | `ds_read_b128 v[138:141], v41` | `v_add_u32_e32 v41, 0x1cd60, v12` | ds_read | 4.0 | 0.0 | 4.0 | 2.5% |
| 741 | `ds_read_b128 v[98:101], v41 offset:64` | `ds_read_b128 v[94:97], v41` | ds_read | 0.0 | 4.0 | 4.0 | 2.5% |
| 742 | `ds_read_b128 v[90:93], v41 offset:4224` | `ds_read_b128 v[98:101], v41 offset:64` | ds_read | 0.0 | 4.0 | 4.0 | 2.5% |
| 925 | `ds_read_b128 v[138:141], v41` | `v_add_u32_e32 v41, 0x1dde0, v12` | ds_read | 4.0 | 0.0 | 4.0 | 2.5% |
| 974 | `ds_read_b128 v[102:105], v41 offset:64` | `ds_read_b128 v[98:101], v41` | ds_read | 0.0 | 4.0 | 4.0 | 2.5% |
| 975 | `ds_read_b128 v[90:93], v41 offset:4224` | `ds_read_b128 v[102:105], v41 offset:64` | ds_read | 0.0 | 4.0 | 4.0 | 2.5% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 611 | 668 `s_waitcnt lgkmcnt(7)` | 29 | 29 | 348 | 72 | yes | 25 | 4 |
| 671 | 675 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 3 | 2 | 36 | 0 | yes | 0 | 4 |
| 676 | 676 `s_barrier` | 0 | 1 | 0 | 0 | yes | 12 | 20 |
| 677 | 737 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 28 | 33 | 328 | 87 | yes | 49 | 4 |
| 738 | 738 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 739 | 793 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 24 | 31 | 280 | 75 | yes | 55 | 4 |
| 796 | 844 `s_waitcnt lgkmcnt(7)` | 23 | 26 | 276 | 56 | yes | 7 | 4 |
| 851 | 906 `s_waitcnt lgkmcnt(7)` | 29 | 27 | 348 | 64 | yes | 8 | 4 |
| 909 | 913 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 3 | 2 | 36 | 0 | yes | 0 | 4 |
| 914 | 914 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 915 | 970 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 28 | 28 | 328 | 67 | yes | 29 | 4 |
| 971 | 971 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 972 | 1026 `s_waitcnt vmcnt(19) lgkmcnt(0)` | 24 | 31 | 280 | 75 | yes | 56 | 4 |
| 1029 | 604 `s_waitcnt lgkmcnt(7)` | 23 | 33 | 276 | 84 | yes | 48 | 20 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 604 | `s_waitcnt lgkmcnt(7)` | lgkmcnt | 0.98 | 20 | 396 | 23.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 610 | 16 | {3: 111, 1: 62, 2: 16} |
| 676 | 26 | {3: 120, 1: 69} |
| 738 | 13 | {3: 127, 1: 62} |
| 795 | 16 | {1: 49, 0: 48, 3: 47, 2: 45} |
| 850 | 13 | {3: 111, 1: 63, 2: 15} |
| 914 | 13 | {3: 126, 1: 63} |
| 971 | 13 | {3: 126, 1: 63} |
| 1028 | 15 | {2: 49, 1: 48, 0: 47, 3: 45} |

### 256x240_4096x3840x8192

`v9_tile/tmp/tile_n/att_r2/sweep_att/256x240_4096x3840x8192/ui_output_agent_38965_dispatch_60`: 12 waves, 63 iterations each. Per iteration and wave: 4165 cycles, 325 of them MFMA idle (loop eff 92.2%); MFMA stall 1416 pipe-full, 3 pipe-empty; unattributed 0.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| issue | 128 | 39.5% |
| dependency hole | 80 | 24.6% |
| barrier | 30 | 9.1% |
| lds: ds_read stall | 29 | 9.0% |
| other stall | 25 | 7.7% |
| memory: vmcnt wait | 20 | 6.1% |
| memory: buffer_load issue stall | 10 | 3.1% |
| mfma stall, pipe empty | 3 | 0.9% |
| lds: lgkmcnt wait | 0 | 0.0% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 658 | `s_waitcnt vmcnt(20) lgkmcnt(0)` | `v_accvgpr_write_b32 a211, v0` | s_waitcnt vmcnt+lgkmcnt | 0.0 | 19.7 | 0.0 | 6.1% |
| 798 | `ds_read_b128 v[112:115], v0` | `v_add_u32_e32 v0, 0x128c0, v14` | ds_read | 8.0 | 2.0 | 4.0 | 4.3% |
| 990 | `ds_read_b128 v[192:195], v0` | `v_add_u32_e32 v0, 0x16aa0, v14` | ds_read | 8.0 | 2.0 | 4.0 | 4.3% |
| 1052 | `ds_read_b128 v[120:123], v0` | `v_add_u32_e32 v0, 0x107c0, v14` | ds_read | 8.0 | 2.0 | 4.0 | 4.3% |
| 732 | `ds_read_b128 v[192:195], v0` | `v_add_u32_e32 v0, 0x149a0, v14` | ds_read | 8.0 | 2.0 | 4.0 | 4.3% |
| 659 | `v_mfma_f32_16x16x32_bf16 a[236:239], v[120:123], v[92:95], a` | `s_waitcnt vmcnt(20) lgkmcnt(0)` | mfma | 11.8 | 0.0 | 0.0 | 3.6% |
| 741 | `ds_read_b128 v[184:187], v0` | `v_add_u32_e32 v0, 0x1cd60, v14` | ds_read | 4.0 | 0.0 | 4.0 | 2.5% |
| 746 | `ds_read_b128 v[176:179], v0` | `v_add_u32_e32 v0, 0x1ee40, v13` | ds_read | 4.0 | 0.0 | 4.0 | 2.5% |
| 749 | `ds_read_b128 v[180:183], v0` | `v_add_u32_e32 v0, 0x1ee40, v12` | ds_read | 4.0 | 0.0 | 4.0 | 2.5% |
| 809 | `ds_read_b128 v[96:99], v0` | `v_add_u32_e32 v0, 0x1ac80, v14` | ds_read | 4.0 | 0.0 | 4.0 | 2.5% |
| 858 | `ds_read_b128 v[52:55], v1 offset:16960` | `ds_read_b128 v[56:59], v1 offset:16896` | ds_read | 0.0 | 4.0 | 4.0 | 2.5% |
| 859 | `ds_read_b128 v[48:51], v1 offset:17152` | `ds_read_b128 v[52:55], v1 offset:16960` | ds_read | 0.0 | 4.0 | 4.0 | 2.5% |
| 999 | `ds_read_b128 v[180:183], v0` | `v_add_u32_e32 v0, 0x1dde0, v14` | ds_read | 4.0 | 0.0 | 4.0 | 2.5% |
| 1004 | `ds_read_b128 v[176:179], v0` | `v_add_u32_e32 v0, 0x1f640, v13` | ds_read | 4.0 | 0.0 | 4.0 | 2.5% |
| 1007 | `ds_read_b128 v[184:187], v0` | `v_add_u32_e32 v0, 0x1f640, v12` | ds_read | 4.0 | 0.0 | 4.0 | 2.5% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 659 | 660 `s_barrier` | 1 | 1 | 12 | 0 | yes | 15 | 14 |
| 661 | 723 `s_waitcnt lgkmcnt(11)` | 31 | 32 | 372 | 88 | yes | 16 | 4 |
| 730 | 794 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 29 | 36 | 348 | 100 | yes | 57 | 4 |
| 797 | 855 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 27 | 32 | 324 | 72 | yes | 28 | 4 |
| 856 | 856 `s_barrier` | 0 | 1 | 0 | 0 | yes | 5 | 13 |
| 857 | 919 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 28 | 35 | 328 | 83 | yes | 29 | 4 |
| 922 | 982 `s_waitcnt lgkmcnt(11)` | 31 | 30 | 372 | 80 | yes | 9 | 4 |
| 989 | 1048 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 29 | 31 | 348 | 80 | yes | 45 | 4 |
| 1051 | 1109 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 27 | 32 | 324 | 72 | yes | 28 | 4 |
| 1110 | 1110 `s_barrier` | 0 | 1 | 0 | 0 | yes | 6 | 14 |
| 1111 | 658 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 28 | 42 | 328 | 111 | yes | 70 | 20 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 658 | `s_waitcnt vmcnt(20) lgkmcnt(0)` | vmcnt | 0.98 | 20 | 2653 | 154.0 |
| 658 | `s_waitcnt vmcnt(20) lgkmcnt(0)` | lgkmcnt | 0.98 | 20 | 335 | 19.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 660 | 13 | {3: 126, 1: 62, 2: 1} |
| 729 | 15 | {3: 126, 1: 62, 2: 1} |
| 796 | 12 | {3: 126, 1: 63} |
| 856 | 12 | {3: 126, 1: 63} |
| 921 | 12 | {3: 126, 1: 63} |
| 988 | 12 | {3: 126, 1: 63} |
| 1050 | 12 | {3: 126, 1: 63} |
| 1110 | 13 | {3: 126, 1: 63} |

### 256x256_4096x4096x8192

`v9_tile/tmp/tile_n/att_r2/sweep_att/256x256_4096x4096x8192/ui_output_agent_17729_dispatch_58`: 12 waves, 63 iterations each. Per iteration and wave: 4291 cycles, 195 of them MFMA idle (loop eff 95.5%); MFMA stall 1652 pipe-full, 1 pipe-empty; unattributed 0.

| exposed as | cycles / iter | share of idle |
|---|---|---|
| issue | 70 | 35.8% |
| dependency hole | 48 | 24.6% |
| lds: lgkmcnt wait | 20 | 10.1% |
| barrier | 19 | 10.0% |
| other stall | 17 | 8.6% |
| memory: buffer_load issue stall | 12 | 6.3% |
| lds: ds_read stall | 8 | 4.3% |
| mfma stall, pipe empty | 1 | 0.3% |
| memory: vmcnt wait | 0 | 0.0% |

Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, waiting on the previous one):

| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |
|---|---|---|---|---|---|---|---|
| 634 | `s_waitcnt lgkmcnt(7)` | `v_accvgpr_write_b32 a195, v1` | s_waitcnt lgkmcnt | 0.0 | 19.7 | 0.0 | 10.1% |
| 774 | `ds_read_b128 v[64:67], v39` | `v_add_u32_e32 v39, 0x149c0, v1` | ds_read | 8.0 | 2.0 | 4.0 | 7.2% |
| 967 | `ds_read_b128 v[144:147], v39` | `v_add_u32_e32 v39, 0x1cda0, v1` | ds_read | 8.0 | 2.0 | 4.0 | 7.2% |
| 1030 | `ds_read_b128 v[96:99], v39` | `v_add_u32_e32 v39, 0x107c0, v1` | ds_read | 8.0 | 2.0 | 4.0 | 7.2% |
| 707 | `ds_read_b128 v[144:147], v39` | `v_add_u32_e32 v39, 0x18ba0, v1` | ds_read | 8.0 | 2.0 | 4.0 | 7.2% |
| 635 | `v_mfma_f32_16x16x32_bf16 a[252:255], v[96:99], v[88:91], a[2` | `s_waitcnt lgkmcnt(7)` | mfma | 11.8 | 0.0 | 0.0 | 6.1% |
| 1160 | `s_cbranch_scc1 64649` | `s_cmpk_lt_u32 s18, 0x7c` | other | 4.0 | 0.0 | 4.0 | 4.1% |
| 677 | `buffer_load_dwordx4 v4, s[0:3], 0 offen lds` | `v_mfma_f32_16x16x32_bf16 a[216:219], v[8` | buffer_load | 0.0 | 6.5 | 0.2 | 3.4% |
| 704 | `s_barrier` | `v_mfma_f32_16x16x32_bf16 a[184:187], v[7` | s_barrier | 0.0 | 5.7 | 0.0 | 2.9% |
| 685 | `buffer_load_dwordx4 v5, s[0:3], 0 offen lds` | `v_mfma_f32_16x16x32_bf16 a[208:211], v[6` | buffer_load | 0.0 | 4.5 | 0.2 | 2.4% |
| 1095 | `ds_read_b128 v[88:91], v0` | `s_barrier` | ds_read | 0.0 | 0.1 | 4.0 | 2.1% |
| 839 | `ds_read_b128 v[136:139], v0 offset:16896` | `s_barrier` | ds_read | 0.0 | 0.1 | 4.0 | 2.1% |
| 904 | `ds_read_b128 v[128:131], v0 offset:50656` | `s_barrier` | ds_read | 0.0 | 0.0 | 4.0 | 2.1% |
| 641 | `ds_read_b128 v[128:131], v0 offset:33760` | `s_barrier` | ds_read | 0.0 | 0.0 | 4.0 | 2.1% |
| 706 | `v_add_u32_e32 v39, 0x18ba0, v1` | `v_add_u32_e32 v1, 0, v38` | other | 0.0 | 0.0 | 4.0 | 2.1% |

Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle cycles left out):

| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |
|---|---|---|---|---|---|---|---|---|
| 635 | 636 `s_waitcnt lgkmcnt(6)` | 1 | 1 | 12 | 0 | yes | 12 | 4 |
| 641 | 698 `s_waitcnt lgkmcnt(7)` | 29 | 29 | 348 | 72 | yes | 20 | 4 |
| 703 | 704 `s_barrier` | 1 | 1 | 12 | 0 | yes | 6 | 16 |
| 705 | 766 `s_waitcnt lgkmcnt(7)` | 29 | 33 | 348 | 88 | yes | 30 | 4 |
| 773 | 836 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 31 | 33 | 372 | 68 | yes | 18 | 4 |
| 839 | 897 `s_waitcnt lgkmcnt(7)` | 29 | 30 | 348 | 64 | yes | 6 | 4 |
| 904 | 959 `s_waitcnt lgkmcnt(7)` | 29 | 27 | 348 | 64 | yes | 6 | 4 |
| 966 | 1022 `s_waitcnt lgkmcnt(7)` | 29 | 28 | 348 | 68 | yes | 18 | 4 |
| 1029 | 1092 `s_waitcnt vmcnt(20) lgkmcnt(0)` | 31 | 33 | 372 | 68 | yes | 19 | 4 |
| 1095 | 634 `s_waitcnt lgkmcnt(7)` | 29 | 38 | 348 | 92 | yes | 46 | 20 |

Stalled waits (>= 8 cycles): distance from the awaited load's issue:

| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |
|---|---|---|---|---|---|---|
| 634 | `s_waitcnt lgkmcnt(7)` | lgkmcnt | 0.98 | 20 | 490 | 29.0 |

Barrier arrival skew across a CU's waves:

| code index | mean skew (cycles) | last to arrive (SIMD: count) |
|---|---|---|
| 640 | 11 | {1: 126, 3: 62, 2: 1} |
| 704 | 18 | {1: 138, 3: 50, 2: 1} |
| 772 | 11 | {1: 127, 3: 62} |
| 838 | 11 | {1: 126, 3: 63} |
| 903 | 11 | {1: 126, 3: 63} |
| 965 | 11 | {1: 126, 3: 63} |
| 1028 | 11 | {1: 127, 3: 62} |
| 1094 | 11 | {1: 126, 3: 63} |

