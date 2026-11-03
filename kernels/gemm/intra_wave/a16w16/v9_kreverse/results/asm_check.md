# v9_kreverse asm check (8192x8192x8192, bf16)

| build | VGPRs / spills | SGPRs | instruction lines changed vs v9 | hot-loop instructions | hot-loop mix (v_mfma, buffer_load, ds_read, v_*, s_*) | result |
|---|---|---|---|---|---|---|
| v9_beyond_hotloop | 456 / 0 | - | - | 423 | 256, 32, 64, 0, 43 | reference |
| K_ORDER=0 | 456 / 0 | - | 0 | 423 | 256, 32, 64, 0, 43 | identical |
| K_ORDER=1 | 456 / 0 | - | 481 | 423 | 256, 32, 64, 0, 43 | same loop size and mix |
| K_ORDER=2 | 456 / 0 | - | 470 | 423 | 256, 32, 64, 0, 43 | same loop size and mix |

Hot-loop lines that differ from v9, K_ORDER=1:

```
-s_and_b32 s1, s52, 0xffff
-s_mov_b32 s0, s51
-s_mov_b32 m0, s11
+s_and_b32 s1, s11, 0xffff
+s_mov_b32 s0, s10
+s_mov_b32 m0, s27
-s_mov_b32 m0, s17
+s_mov_b32 m0, s28
-s_mov_b32 m0, s20
+s_mov_b32 m0, s29
-s_mov_b32 m0, s21
+s_mov_b32 m0, s30
-s_mov_b32 m0, s22
+s_mov_b32 m0, s31
-s_mov_b32 m0, s23
+s_mov_b32 m0, s33
-s_mov_b32 m0, s24
+s_mov_b32 m0, s34
-s_mov_b32 m0, s25
+s_mov_b32 m0, s35
-s_mov_b32 m0, s26
+s_mov_b32 m0, s36
-s_mov_b32 m0, s27
+s_mov_b32 m0, s37
-s_mov_b32 m0, s28
+s_mov_b32 m0, s38
-s_mov_b32 m0, s29
+s_mov_b32 m0, s39
-s_mov_b32 m0, s30
+s_mov_b32 m0, s40
-s_mov_b32 m0, s31
+s_mov_b32 m0, s41
-s_mov_b32 m0, s33
+s_mov_b32 m0, s42
-s_mov_b32 m0, s34
+s_mov_b32 m0, s43
-s_mov_b32 m0, s35
+s_mov_b32 m0, s44
-s_mov_b32 m0, s36
+s_mov_b32 m0, s45
-s_mov_b32 m0, s37
+s_mov_b32 m0, s46
-s_mov_b32 m0, s38
+s_mov_b32 m0, s47
-s_mov_b32 m0, s39
+s_mov_b32 m0, s48
-s_mov_b32 m0, s40
+s_mov_b32 m0, s49
-s_mov_b32 m0, s41
+s_mov_b32 m0, s50
-s_mov_b32 m0, s42
+s_mov_b32 m0, s51
-s_mov_b32 m0, s43
+s_mov_b32 m0, s52
-s_mov_b32 m0, s44
+s_mov_b32 m0, s53
-s_mov_b32 m0, s45
+s_mov_b32 m0, s54
-s_mov_b32 m0, s46
+s_mov_b32 m0, s55
-s_mov_b32 m0, s47
+s_mov_b32 m0, s56
-s_mov_b32 m0, s48
+s_mov_b32 m0, s57
-s_mov_b32 m0, s49
+s_mov_b32 m0, s58
-s_mov_b32 m0, s50
+s_mov_b32 m0, s59
-s_add_u32 s51, s51, 0x100
-s_addc_u32 s52, s52, 0
-s_add_u32 s8, s8, 0x100
-s_addc_u32 s9, s9, 0
-s_add_i32 s16, s16, 2
+s_add_u32 s10, s10, s20
+s_addc_u32 s11, s11, s21
+s_add_u32 s8, s8, s20
+s_addc_u32 s9, s9, s21
+s_add_i32 s26, s26, 2
-s_cmpk_lt_u32 s16, 0x7c
+s_cmpk_lt_u32 s26, 0x7c
```

Hot-loop lines that differ from v9, K_ORDER=2:

```
-s_and_b32 s1, s52, 0xffff
-s_mov_b32 s0, s51
-s_mov_b32 m0, s11
+s_and_b32 s1, s51, 0xffff
+s_mov_b32 s0, s50
+s_mov_b32 m0, s9
-s_mov_b32 m0, s17
+s_mov_b32 m0, s11
-s_mov_b32 m0, s20
+s_mov_b32 m0, s17
-s_mov_b32 m0, s21
+s_mov_b32 m0, s18
-s_and_b32 s5, s9, 0xffff
-s_mov_b32 s4, s8
-s_mov_b32 m0, s22
+s_and_b32 s5, s49, 0xffff
+s_mov_b32 s4, s48
+s_mov_b32 m0, s19
-s_mov_b32 m0, s23
+s_mov_b32 m0, s20
-s_mov_b32 m0, s24
+s_mov_b32 m0, s21
-s_mov_b32 m0, s25
+s_mov_b32 m0, s22
-s_mov_b32 m0, s26
+s_mov_b32 m0, s23
-s_mov_b32 m0, s27
+s_mov_b32 m0, s24
-s_mov_b32 m0, s28
+s_mov_b32 m0, s25
-s_mov_b32 m0, s29
+s_mov_b32 m0, s26
-s_mov_b32 m0, s30
+s_mov_b32 m0, s27
-s_mov_b32 m0, s31
+s_mov_b32 m0, s28
-s_mov_b32 m0, s33
+s_mov_b32 m0, s29
-s_mov_b32 m0, s34
+s_mov_b32 m0, s30
-s_mov_b32 m0, s35
+s_mov_b32 m0, s31
-s_mov_b32 m0, s36
+s_mov_b32 m0, s33
-s_mov_b32 m0, s37
+s_mov_b32 m0, s34
-s_mov_b32 m0, s38
+s_mov_b32 m0, s35
-s_mov_b32 m0, s39
+s_mov_b32 m0, s36
-s_mov_b32 m0, s40
+s_mov_b32 m0, s37
-s_mov_b32 m0, s41
+s_mov_b32 m0, s38
-s_mov_b32 m0, s42
+s_mov_b32 m0, s39
-s_mov_b32 m0, s43
+s_mov_b32 m0, s40
-s_mov_b32 m0, s44
+s_mov_b32 m0, s41
-s_mov_b32 m0, s45
+s_mov_b32 m0, s42
-s_mov_b32 m0, s46
+s_mov_b32 m0, s43
-s_mov_b32 m0, s47
+s_mov_b32 m0, s44
-s_mov_b32 m0, s48
+s_mov_b32 m0, s45
-s_mov_b32 m0, s49
+s_mov_b32 m0, s46
-s_mov_b32 m0, s50
+s_mov_b32 m0, s47
-s_add_u32 s51, s51, 0x100
-s_addc_u32 s52, s52, 0
-s_add_u32 s8, s8, 0x100
-s_addc_u32 s9, s9, 0
-s_add_i32 s16, s16, 2
+s_add_u32 s48, s48, 0xffffff00
+s_addc_u32 s49, s49, -1
+s_add_u32 s50, s50, 0xffffff00
+s_addc_u32 s51, s51, -1
+s_add_i32 s52, s52, 2
-s_cmpk_lt_u32 s16, 0x7c
+s_cmpk_lt_u32 s52, 0x7c
```

PASS
