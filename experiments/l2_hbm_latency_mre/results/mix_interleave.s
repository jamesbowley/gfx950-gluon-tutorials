	.amdgcn_target "amdgpu9.50-amd-amdhsa-unknown-gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	mix_kernel                      ; -- Begin function mix_kernel
	.p2align	8
	.type	mix_kernel,@function
mix_kernel:                             ; @mix_kernel
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.18:
	.file	1 "/home/jbowley/repos/gfx950-gluon-tutorials/experiments/l2_hbm_latency_mre" "mre.py"
	.loc	1 211 0 prologue_end            ; mre.py:211:0
	s_load_dwordx8 s[8:15], s[4:5], 0x0
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.19:
.LBB0_0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	v_and_b32_e32 v4, 0x3ff, v0
.Ltmp1:
	.loc	1 128 5 is_stmt 1               ; mre.py:128:5 @[ mre.py:218:11 ]
	v_and_b32_e32 v2, 63, v0
.Ltmp2:
	.loc	1 255 5                         ; mre.py:255:5
	v_readfirstlane_b32 s0, v4
	s_bfe_u32 s17, s0, 0x20006
.Ltmp3:
	.loc	1 128 5                         ; mre.py:128:5 @[ mre.py:218:11 ]
	v_lshl_or_b32 v5, s17, 6, v2
	v_add_u32_e32 v3, 0, v5
	v_mov_b32_e32 v0, 0
	.loc	1 129 5                         ; mre.py:129:5 @[ mre.py:218:11 ]
	v_add_u32_e32 v1, 0x10000, v3
	.loc	1 128 5                         ; mre.py:128:5 @[ mre.py:218:11 ]
	ds_write_b8 v3, v0
	.loc	1 129 5                         ; mre.py:129:5 @[ mre.py:218:11 ]
	ds_write_b8 v1, v0
.Ltmp4:
	.loc	1 72 12                         ; mre.py:72:12 @[ mre.py:78:19 @[ mre.py:111:11 @[ mre.py:220:19 ] ] ]
	;;#ASMSTART
	s_getreg_b32 s0, hwreg(HW_REG_XCC_ID)
	;;#ASMEND
.Ltmp5:
	.loc	1 111 11                        ; mre.py:111:11 @[ mre.py:220:19 ]
	s_and_b32 s18, s0, 15
.Ltmp6:
	.loc	1 72 12                         ; mre.py:72:12 @[ mre.py:78:19 @[ mre.py:112:10 @[ mre.py:220:19 ] ] ]
	;;#ASMSTART
	s_getreg_b32 s0, hwreg(HW_REG_HW_ID)
	;;#ASMEND
.Ltmp7:
	.loc	1 115 45                        ; mre.py:115:45 @[ mre.py:220:19 ]
	s_bfe_u32 s20, s0, 0x3000d
	.loc	1 115 61 is_stmt 0              ; mre.py:115:61 @[ mre.py:220:19 ]
	s_bfe_u32 s19, s0, 0x40008
	.loc	1 116 20 is_stmt 1              ; mre.py:116:20 @[ mre.py:220:19 ]
	s_lshl_b32 s0, s16, 4
	.loc	1 116 14 is_stmt 0              ; mre.py:116:14 @[ mre.py:220:19 ]
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s2, s12, s0
	.loc	1 116 45                        ; mre.py:116:45 @[ mre.py:220:19 ]
	v_cmp_gt_u32_e32 vcc, 6, v5
	.loc	1 116 14                        ; mre.py:116:14 @[ mre.py:220:19 ]
	s_addc_u32 s3, s13, s1
	.loc	1 116 5                         ; mre.py:116:5 @[ mre.py:220:19 ]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	.loc	1 115 25 is_stmt 1              ; mre.py:115:25 @[ mre.py:220:19 ]
	v_mov_b32_e32 v7, s19
	v_mov_b32_e32 v8, s20
	.loc	1 115 34 is_stmt 0              ; mre.py:115:34 @[ mre.py:220:19 ]
	v_cmp_eq_u32_e64 s[0:1], 4, v5
	.loc	1 114 18 is_stmt 1              ; mre.py:114:18 @[ mre.py:220:19 ]
	v_subrev_co_u32_e32 v6, vcc, 1, v5
	.loc	1 115 25                        ; mre.py:115:25 @[ mre.py:220:19 ]
	s_nop 0
	v_cndmask_b32_e64 v7, v7, v8, s[0:1]
	.loc	1 114 84                        ; mre.py:114:84 @[ mre.py:220:19 ]
	v_mov_b32_e32 v8, s18
	.loc	1 115 9                         ; mre.py:115:9 @[ mre.py:220:19 ]
	v_cmp_eq_u32_e64 s[0:1], 3, v5
	.loc	1 114 9                         ; mre.py:114:9 @[ mre.py:220:19 ]
	v_mov_b32_e32 v1, s16
	.loc	1 114 84 is_stmt 0              ; mre.py:114:84 @[ mre.py:220:19 ]
	s_nop 0
	v_cndmask_b32_e64 v7, v7, v8, s[0:1]
	.loc	1 114 34                        ; mre.py:114:34 @[ mre.py:220:19 ]
	v_cmp_lt_u32_e64 s[0:1], 1, v6
	s_nop 1
	v_cndmask_b32_e64 v6, 0, v7, s[0:1]
	.loc	1 114 9                         ; mre.py:114:9 @[ mre.py:220:19 ]
	v_cndmask_b32_e32 v1, v6, v1, vcc
	.loc	1 116 14 is_stmt 1              ; mre.py:116:14 @[ mre.py:220:19 ]
	v_lshlrev_b32_e32 v6, 2, v5
	.loc	1 116 5 is_stmt 0               ; mre.py:116:5 @[ mre.py:220:19 ]
	global_store_dword v6, v1, s[2:3]
.LBB0_2:
	.loc	1 0 5                           ; mre.py:0:5
	s_or_b64 exec, exec, s[6:7]
	s_load_dwordx2 s[0:1], s[4:5], 0x20
	s_load_dword s12, s[4:5], 0x28
	.loc	1 117 82 is_stmt 1              ; mre.py:117:82 @[ mre.py:220:19 ]
	v_cmp_eq_u32_e32 vcc, 0, v2
.Ltmp8:
	.loc	1 72 12                         ; mre.py:72:12 @[ mre.py:117:47 @[ mre.py:220:19 ] ]
	;;#ASMSTART
	s_getreg_b32 s13, hwreg(HW_REG_HW_ID)
	;;#ASMEND
.Ltmp9:
	.loc	1 117 5                         ; mre.py:117:5 @[ mre.py:220:19 ]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_4
; %bb.3:
	.loc	1 0 5 is_stmt 0                 ; mre.py:0:5
	s_lshl_b32 s21, s17, 2
	.loc	1 117 5                         ; mre.py:117:5 @[ mre.py:220:19 ]
	v_mov_b32_e32 v0, s21
	v_mov_b32_e32 v1, s13
	global_store_dword v0, v1, s[2:3] offset:32
.Ltmp10:
.LBB0_4:
	.loc	1 0 5                           ; mre.py:0:5
	s_or_b64 exec, exec, s[6:7]
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u32 s18, s0
	s_cselect_b64 s[2:3], -1, 0
	.loc	1 222 5 is_stmt 1               ; mre.py:222:5
	s_and_b64 s[6:7], s[2:3], exec
	s_cselect_b32 s0, 1, 0
	.loc	1 221 10                        ; mre.py:221:10
	s_cmp_gt_i32 s16, -1
	.loc	1 222 5                         ; mre.py:222:5
	s_cselect_b32 s6, 1, 0
	.loc	1 222 8 is_stmt 0               ; mre.py:222:8
	s_cmp_eq_u32 s15, 1
	.loc	1 222 5                         ; mre.py:222:5
	s_cselect_b32 s13, s0, s6
	s_cmp_eq_u32 s20, s1
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_eq_u32 s19, s12
	s_cselect_b64 s[6:7], -1, 0
	s_and_b64 s[0:1], s[0:1], s[6:7]
	s_and_b64 s[0:1], s[2:3], s[0:1]
	.loc	1 224 5 is_stmt 1               ; mre.py:224:5
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s0, 1, 0
	.loc	1 224 8 is_stmt 0               ; mre.py:224:8
	s_cmp_eq_u32 s15, 2
	.loc	1 224 5                         ; mre.py:224:5
	s_cselect_b32 s0, s0, s13
	.loc	1 231 5 is_stmt 1               ; mre.py:231:5
	s_bitcmp0_b32 s0, 0
	s_mov_b32 s13, 0
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v0, 0
	s_cbranch_scc1 .LBB0_15
; %bb.5:                                ; %.preheader
	.loc	1 0 5 is_stmt 0                 ; mre.py:0:5
	s_load_dword s34, s[4:5], 0x2c
	.loc	1 228 15 is_stmt 1              ; mre.py:228:15
	s_lshl_b32 s0, s18, 19
	s_add_u32 s15, s8, s0
	.loc	1 227 25                        ; mre.py:227:25
	v_lshlrev_b32_e32 v1, 4, v4
	.loc	1 228 15                        ; mre.py:228:15
	s_addc_u32 s18, s9, 0
	.loc	1 227 13                        ; mre.py:227:13
	v_lshlrev_b32_e32 v0, 4, v5
	.loc	1 227 25 is_stmt 0              ; mre.py:227:25
	v_and_b32_e32 v1, 16, v1
	.loc	1 227 13                        ; mre.py:227:13
	s_movk_i32 s0, 0xfe0
	s_lshl_b32 s33, s17, 10
	v_and_or_b32 v1, v0, s0, v1
	s_add_i32 s35, s33, 0x18000
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	s_mov_b32 s76, 0
.LBB0_6:                                ; =>This Inner Loop Header: Depth=1
	.loc	1 234 64 is_stmt 1              ; mre.py:234:64
	s_lshl_b32 s12, s76, 11
	.loc	1 234 54 is_stmt 0              ; mre.py:234:54
	s_lshl_b64 s[0:1], s[12:13], 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s12, s18, s1
	.loc	1 234 13                        ; mre.py:234:13
	s_and_b32 s1, s12, 0xffff
	s_mov_b32 m0, s35
	.loc	1 234 54                        ; mre.py:234:54
	s_add_u32 s4, s0, 0x1000
	.loc	1 234 13                        ; mre.py:234:13
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 234 54                        ; mre.py:234:54
	s_addc_u32 s1, s12, 0
	.loc	1 234 13                        ; mre.py:234:13
	s_and_b32 s5, s1, 0xffff
	.loc	1 234 54                        ; mre.py:234:54
	s_add_u32 s20, s0, 0x2000
	s_addc_u32 s1, s12, 0
	.loc	1 234 13                        ; mre.py:234:13
	s_and_b32 s21, s1, 0xffff
	.loc	1 234 54                        ; mre.py:234:54
	s_add_u32 s24, s0, 0x3000
	s_addc_u32 s1, s12, 0
	.loc	1 234 13                        ; mre.py:234:13
	s_and_b32 s25, s1, 0xffff
	.loc	1 234 54                        ; mre.py:234:54
	s_add_u32 s28, s0, 0x4000
	s_addc_u32 s1, s12, 0
	.loc	1 234 13                        ; mre.py:234:13
	s_and_b32 s29, s1, 0xffff
	.loc	1 234 54                        ; mre.py:234:54
	s_add_u32 s36, s0, 0x5000
	s_addc_u32 s1, s12, 0
	.loc	1 234 13                        ; mre.py:234:13
	s_and_b32 s37, s1, 0xffff
	.loc	1 234 54                        ; mre.py:234:54
	s_add_u32 s40, s0, 0x6000
	s_addc_u32 s1, s12, 0
	.loc	1 234 13                        ; mre.py:234:13
	s_and_b32 s41, s1, 0xffff
	.loc	1 234 54                        ; mre.py:234:54
	s_add_u32 s44, s0, 0x7000
	s_addc_u32 s1, s12, 0
	.loc	1 234 13                        ; mre.py:234:13
	s_and_b32 s45, s1, 0xffff
	.loc	1 234 54                        ; mre.py:234:54
	s_add_u32 s48, s0, 0x8000
	s_addc_u32 s1, s12, 0
	.loc	1 234 13                        ; mre.py:234:13
	s_and_b32 s49, s1, 0xffff
	.loc	1 234 54                        ; mre.py:234:54
	s_add_u32 s52, s0, 0x9000
	s_addc_u32 s1, s12, 0
	.loc	1 234 13                        ; mre.py:234:13
	s_and_b32 s53, s1, 0xffff
	.loc	1 234 54                        ; mre.py:234:54
	s_add_u32 s56, s0, 0xa000
	s_addc_u32 s1, s12, 0
	.loc	1 234 13                        ; mre.py:234:13
	s_and_b32 s57, s1, 0xffff
	.loc	1 234 54                        ; mre.py:234:54
	s_add_u32 s60, s0, 0xb000
	s_addc_u32 s1, s12, 0
	.loc	1 234 13                        ; mre.py:234:13
	s_and_b32 s61, s1, 0xffff
	.loc	1 234 54                        ; mre.py:234:54
	s_add_u32 s64, s0, 0xc000
	s_addc_u32 s1, s12, 0
	.loc	1 234 13                        ; mre.py:234:13
	s_and_b32 s65, s1, 0xffff
	.loc	1 234 54                        ; mre.py:234:54
	s_add_u32 s68, s0, 0xd000
	s_addc_u32 s1, s12, 0
	.loc	1 234 13                        ; mre.py:234:13
	s_and_b32 s69, s1, 0xffff
	.loc	1 234 54                        ; mre.py:234:54
	s_add_u32 s72, s0, 0xe000
	s_addc_u32 s1, s12, 0
	.loc	1 234 13                        ; mre.py:234:13
	s_and_b32 s73, s1, 0xffff
	.loc	1 234 54                        ; mre.py:234:54
	s_add_u32 s0, s0, 0xf000
	s_addc_u32 s1, s12, 0
	.loc	1 234 13                        ; mre.py:234:13
	s_mov_b32 s6, s2
	s_mov_b32 s7, s3
	s_mov_b32 s22, s2
	s_mov_b32 s23, s3
	s_mov_b32 s26, s2
	s_mov_b32 s27, s3
	s_mov_b32 s30, s2
	s_mov_b32 s31, s3
	s_mov_b32 s38, s2
	s_mov_b32 s39, s3
	s_mov_b32 s42, s2
	s_mov_b32 s43, s3
	s_mov_b32 s46, s2
	s_mov_b32 s47, s3
	s_mov_b32 s50, s2
	s_mov_b32 s51, s3
	s_mov_b32 s54, s2
	s_mov_b32 s55, s3
	s_mov_b32 s58, s2
	s_mov_b32 s59, s3
	s_mov_b32 s62, s2
	s_mov_b32 s63, s3
	s_mov_b32 s66, s2
	s_mov_b32 s67, s3
	s_mov_b32 s70, s2
	s_mov_b32 s71, s3
	s_mov_b32 s74, s2
	s_mov_b32 s75, s3
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[4:7], 0 offen lds
	buffer_load_dwordx4 v1, s[20:23], 0 offen lds
	buffer_load_dwordx4 v1, s[24:27], 0 offen lds
	buffer_load_dwordx4 v1, s[28:31], 0 offen lds
	buffer_load_dwordx4 v1, s[36:39], 0 offen lds
	buffer_load_dwordx4 v1, s[40:43], 0 offen lds
	buffer_load_dwordx4 v1, s[44:47], 0 offen lds
	buffer_load_dwordx4 v1, s[48:51], 0 offen lds
	buffer_load_dwordx4 v1, s[52:55], 0 offen lds
	buffer_load_dwordx4 v1, s[56:59], 0 offen lds
	buffer_load_dwordx4 v1, s[60:63], 0 offen lds
	buffer_load_dwordx4 v1, s[64:67], 0 offen lds
	buffer_load_dwordx4 v1, s[68:71], 0 offen lds
	buffer_load_dwordx4 v1, s[72:75], 0 offen lds
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 233 9 is_stmt 1               ; mre.py:233:9
	s_add_i32 s76, s76, 16
	s_cmpk_lg_i32 s76, 0x80
	s_cbranch_scc1 .LBB0_6
; %bb.7:
.Ltmp11:
	.loc	1 206 5                         ; mre.py:206:5 @[ mre.py:235:9 ]
	v_mov_b32_e32 v5, 0
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
.Ltmp12:
	.loc	1 237 9                         ; mre.py:237:9
	s_cmp_lt_i32 s14, 1
	s_cbranch_scc1 .LBB0_14
; %bb.8:                                ; %.lr.ph
	.loc	1 229 48                        ; mre.py:229:48
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s0, s16, s34
	s_lshl_b32 s0, s0, 11
	.loc	1 229 16 is_stmt 0              ; mre.py:229:16
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s8, s0
	s_addc_u32 s1, s9, s1
	s_add_u32 s8, s0, 0x400000
	s_addc_u32 s9, s1, 0
	s_add_i32 s13, s33, 0x19000
	.loc	1 236 15 is_stmt 1              ; mre.py:236:15
	s_mul_i32 s19, s19, 7
	s_mov_b32 s12, 0
	.loc	1 237 9                         ; mre.py:237:9
	s_cmp_eq_u32 s14, 1
	s_mov_b64 s[0:1], -1
	s_cbranch_scc1 .LBB0_12
; %bb.9:                                ; %.lr.ph.new
	.loc	1 0 9 is_stmt 0                 ; mre.py:0:9
	s_mov_b32 s5, 0
	.loc	1 237 9                         ; mre.py:237:9
	s_mov_b32 s2, 0x7ffffffe
	s_and_b32 s12, s14, 0x7ffffffe
	s_mov_b32 s3, 0x27000
	v_mov_b32_e32 v5, 0
	s_mov_b32 s6, s5
	s_mov_b32 s20, 0
.LBB0_10:                               ; =>This Inner Loop Header: Depth=1
	.loc	1 239 33 is_stmt 1              ; mre.py:239:33
	s_lshl_b32 s4, s19, 11
	.loc	1 239 23 is_stmt 0              ; mre.py:239:23
	s_lshl_b64 s[0:1], s[4:5], 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, s1
	.loc	1 239 33                        ; mre.py:239:33
	s_add_i32 s7, s4, 0x8800
	.loc	1 246 21 is_stmt 1              ; mre.py:246:21
	s_mov_b32 m0, s13
	s_and_b32 s1, s1, 0xffff
	.loc	1 239 33                        ; mre.py:239:33
	s_and_b32 s7, s7, 0x3f800
	.loc	1 246 21                        ; mre.py:246:21
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 239 23                        ; mre.py:239:23
	s_lshl_b32 s0, s7, 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, 0
	.loc	1 240 24                        ; mre.py:240:24
	s_ashr_i32 s7, s6, 31
	.loc	1 246 21                        ; mre.py:246:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 240 24                        ; mre.py:240:24
	s_lshl_b64 s[22:23], s[6:7], 1
	s_add_u32 s7, s8, s22
	s_addc_u32 s21, s9, s23
	.loc	1 246 21                        ; mre.py:246:21
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 24                        ; mre.py:240:24
	s_add_u32 s0, s7, 0x2000
	s_addc_u32 s1, s21, 0
	.loc	1 248 21                        ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 24                        ; mre.py:240:24
	s_add_u32 s0, s7, 0x3000
	s_addc_u32 s1, s21, 0
	.loc	1 239 33                        ; mre.py:239:33
	s_add_i32 s22, s4, 0x22000
	.loc	1 248 21                        ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 239 33                        ; mre.py:239:33
	s_and_b32 s22, s22, 0x3f800
	.loc	1 248 21                        ; mre.py:248:21
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 239 23                        ; mre.py:239:23
	s_lshl_b32 s0, s22, 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, 0
	.loc	1 239 33 is_stmt 0              ; mre.py:239:33
	s_add_i32 s22, s4, 0x2a800
	.loc	1 246 21 is_stmt 1              ; mre.py:246:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 239 33                        ; mre.py:239:33
	s_and_b32 s22, s22, 0x3f800
	.loc	1 246 21                        ; mre.py:246:21
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 239 23                        ; mre.py:239:23
	s_lshl_b32 s0, s22, 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, 0
	.loc	1 246 21                        ; mre.py:246:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 24                        ; mre.py:240:24
	s_add_u32 s0, s7, 0x6000
	s_addc_u32 s1, s21, 0
	.loc	1 248 21                        ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 24                        ; mre.py:240:24
	s_add_u32 s0, s7, 0x7000
	s_addc_u32 s1, s21, 0
	.loc	1 239 33                        ; mre.py:239:33
	s_add_i32 s22, s4, 0x4000
	.loc	1 248 21                        ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 239 33                        ; mre.py:239:33
	s_and_b32 s22, s22, 0x3f800
	.loc	1 248 21                        ; mre.py:248:21
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 239 23                        ; mre.py:239:23
	s_lshl_b32 s0, s22, 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, 0
	.loc	1 239 33 is_stmt 0              ; mre.py:239:33
	s_add_i32 s22, s4, 0xc800
	.loc	1 246 21 is_stmt 1              ; mre.py:246:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 239 33                        ; mre.py:239:33
	s_and_b32 s22, s22, 0x3f800
	.loc	1 246 21                        ; mre.py:246:21
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 239 23                        ; mre.py:239:23
	s_lshl_b32 s0, s22, 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, 0
	.loc	1 246 21                        ; mre.py:246:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 24                        ; mre.py:240:24
	s_add_u32 s0, s7, 0xa000
	s_addc_u32 s1, s21, 0
	.loc	1 248 21                        ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 24                        ; mre.py:240:24
	s_add_u32 s0, s7, 0xb000
	s_addc_u32 s1, s21, 0
	.loc	1 239 33                        ; mre.py:239:33
	s_add_i32 s22, s4, 0x26000
	.loc	1 248 21                        ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 239 33                        ; mre.py:239:33
	s_and_b32 s22, s22, 0x3f800
	.loc	1 248 21                        ; mre.py:248:21
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 239 23                        ; mre.py:239:23
	s_lshl_b32 s0, s22, 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, 0
	.loc	1 239 33 is_stmt 0              ; mre.py:239:33
	s_add_i32 s22, s4, 0x2e800
	.loc	1 246 21 is_stmt 1              ; mre.py:246:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 239 33                        ; mre.py:239:33
	s_and_b32 s22, s22, 0x3f800
	.loc	1 246 21                        ; mre.py:246:21
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 239 23                        ; mre.py:239:23
	s_lshl_b32 s0, s22, 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, 0
	.loc	1 246 21                        ; mre.py:246:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 24                        ; mre.py:240:24
	s_add_u32 s0, s7, 0xe000
	s_addc_u32 s1, s21, 0
	.loc	1 248 21                        ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 24                        ; mre.py:240:24
	s_add_u32 s0, s7, 0xf000
	s_addc_u32 s1, s21, 0
	.loc	1 239 33                        ; mre.py:239:33
	s_add_i32 s22, s4, 0x8000
	.loc	1 248 21                        ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 239 33                        ; mre.py:239:33
	s_and_b32 s22, s22, 0x3f800
	.loc	1 248 21                        ; mre.py:248:21
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 239 23                        ; mre.py:239:23
	s_lshl_b32 s0, s22, 1
.Ltmp13:
	.loc	1 104 5                         ; mre.py:104:5 @[ mre.py:250:13 ]
	;;#ASMSTART
	s_nop 0
	;;#ASMEND
.Ltmp14:
	.loc	1 239 23                        ; mre.py:239:23
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, 0
	.loc	1 239 33 is_stmt 0              ; mre.py:239:33
	s_add_i32 s22, s4, 0x10800
	.loc	1 246 21 is_stmt 1              ; mre.py:246:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 239 33                        ; mre.py:239:33
	s_and_b32 s22, s22, 0x3f800
	.loc	1 246 21                        ; mre.py:246:21
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 239 23                        ; mre.py:239:23
	s_lshl_b32 s0, s22, 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, 0
	.loc	1 246 21                        ; mre.py:246:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 24                        ; mre.py:240:24
	s_add_u32 s0, s7, 0x12000
	s_addc_u32 s1, s21, 0
	.loc	1 248 21                        ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 24                        ; mre.py:240:24
	s_add_u32 s0, s7, 0x13000
	s_addc_u32 s1, s21, 0
	.loc	1 239 33                        ; mre.py:239:33
	s_add_i32 s22, s4, 0x2a000
	.loc	1 248 21                        ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 239 33                        ; mre.py:239:33
	s_and_b32 s22, s22, 0x3f800
	.loc	1 248 21                        ; mre.py:248:21
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 239 23                        ; mre.py:239:23
	s_lshl_b32 s0, s22, 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, 0
	.loc	1 239 33 is_stmt 0              ; mre.py:239:33
	s_add_i32 s22, s4, 0x32800
	.loc	1 246 21 is_stmt 1              ; mre.py:246:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 239 33                        ; mre.py:239:33
	s_and_b32 s22, s22, 0x3f800
	.loc	1 246 21                        ; mre.py:246:21
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 239 23                        ; mre.py:239:23
	s_lshl_b32 s0, s22, 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, 0
	.loc	1 246 21                        ; mre.py:246:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 24                        ; mre.py:240:24
	s_add_u32 s0, s7, 0x16000
	s_addc_u32 s1, s21, 0
	.loc	1 248 21                        ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 24                        ; mre.py:240:24
	s_add_u32 s0, s7, 0x17000
	s_addc_u32 s1, s21, 0
	.loc	1 239 33                        ; mre.py:239:33
	s_add_i32 s22, s4, 0xc000
	.loc	1 248 21                        ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 239 33                        ; mre.py:239:33
	s_and_b32 s22, s22, 0x3f800
	.loc	1 248 21                        ; mre.py:248:21
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 239 23                        ; mre.py:239:23
	s_lshl_b32 s0, s22, 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, 0
	.loc	1 239 33 is_stmt 0              ; mre.py:239:33
	s_add_i32 s22, s4, 0x14800
	.loc	1 246 21 is_stmt 1              ; mre.py:246:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 239 33                        ; mre.py:239:33
	s_and_b32 s22, s22, 0x3f800
	.loc	1 246 21                        ; mre.py:246:21
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 239 23                        ; mre.py:239:23
	s_lshl_b32 s0, s22, 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, 0
	.loc	1 246 21                        ; mre.py:246:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 24                        ; mre.py:240:24
	s_add_u32 s0, s7, 0x1a000
	s_addc_u32 s1, s21, 0
	.loc	1 248 21                        ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 24                        ; mre.py:240:24
	s_add_u32 s0, s7, 0x1b000
	s_addc_u32 s1, s21, 0
	.loc	1 239 33                        ; mre.py:239:33
	s_add_i32 s22, s4, 0x2e000
	.loc	1 248 21                        ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 239 33                        ; mre.py:239:33
	s_and_b32 s22, s22, 0x3f800
	.loc	1 248 21                        ; mre.py:248:21
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 239 23                        ; mre.py:239:23
	s_lshl_b32 s0, s22, 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, 0
	.loc	1 239 33 is_stmt 0              ; mre.py:239:33
	s_add_i32 s4, s4, 0x36800
	.loc	1 246 21 is_stmt 1              ; mre.py:246:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 239 33                        ; mre.py:239:33
	s_and_b32 s4, s4, 0x3f800
	.loc	1 246 21                        ; mre.py:246:21
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 239 23                        ; mre.py:239:23
	s_lshl_b32 s0, s4, 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, 0
	.loc	1 246 21                        ; mre.py:246:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 24                        ; mre.py:240:24
	s_add_u32 s0, s7, 0x1e000
	s_addc_u32 s1, s21, 0
	.loc	1 248 21                        ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 24                        ; mre.py:240:24
	s_add_u32 s0, s7, 0x1f000
	s_addc_u32 s1, s21, 0
	.loc	1 248 21                        ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 249 19                        ; mre.py:249:19
	s_add_i32 s0, s19, 32
.Ltmp15:
	.loc	1 104 5                         ; mre.py:104:5 @[ mre.py:250:13 ]
	;;#ASMSTART
	s_nop 0
	;;#ASMEND
.Ltmp16:
	.loc	1 237 9                         ; mre.py:237:9
	s_add_i32 s20, s20, 2
	s_add_i32 s6, s6, 0x10000
	.loc	1 249 18                        ; mre.py:249:18
	s_and_b32 s19, s0, 0x7f
	.loc	1 237 9                         ; mre.py:237:9
	s_cmp_lg_u32 s12, s20
	s_cbranch_scc1 .LBB0_10
; %bb.11:                               ; %._crit_edge.loopexit.unr-lcssa
	s_bitcmp1_b32 s14, 0
	s_cselect_b64 s[0:1], -1, 0
.LBB0_12:                               ; %Flow
	.loc	1 0 9 is_stmt 0                 ; mre.py:0:9
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB0_14
; %bb.13:                               ; %.epil.preheader
	.loc	1 239 33 is_stmt 1              ; mre.py:239:33
	s_lshl_b32 s4, s19, 11
	.loc	1 239 23 is_stmt 0              ; mre.py:239:23
	s_lshl_b32 s0, s19, 12
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, 0
	.loc	1 246 21 is_stmt 1              ; mre.py:246:21
	s_mov_b32 m0, s13
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 239 33                        ; mre.py:239:33
	s_add_i32 s0, s4, 0x8800
	s_and_b32 s0, s0, 0x3f800
	.loc	1 239 23 is_stmt 0              ; mre.py:239:23
	s_lshl_b32 s0, s0, 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, 0
	.loc	1 246 21 is_stmt 1              ; mre.py:246:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 36                        ; mre.py:240:36
	s_lshl_b32 s0, s12, 15
	.loc	1 240 24 is_stmt 0              ; mre.py:240:24
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s5, s8, s0
	s_addc_u32 s6, s9, s1
	s_add_u32 s0, s5, 0x2000
	s_addc_u32 s1, s6, 0
	.loc	1 248 21 is_stmt 1              ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 24                        ; mre.py:240:24
	s_add_u32 s0, s5, 0x3000
	s_addc_u32 s1, s6, 0
	.loc	1 248 21                        ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 239 33                        ; mre.py:239:33
	s_add_i32 s0, s4, 0x22000
	s_and_b32 s0, s0, 0x3f800
	.loc	1 239 23 is_stmt 0              ; mre.py:239:23
	s_lshl_b32 s0, s0, 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, 0
	.loc	1 246 21 is_stmt 1              ; mre.py:246:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 239 33                        ; mre.py:239:33
	s_add_i32 s0, s4, 0x2a800
	s_and_b32 s0, s0, 0x3f800
	.loc	1 239 23 is_stmt 0              ; mre.py:239:23
	s_lshl_b32 s0, s0, 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, 0
	.loc	1 246 21 is_stmt 1              ; mre.py:246:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 24                        ; mre.py:240:24
	s_add_u32 s0, s5, 0x6000
	s_addc_u32 s1, s6, 0
	.loc	1 248 21                        ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 24                        ; mre.py:240:24
	s_add_u32 s0, s5, 0x7000
	s_addc_u32 s1, s6, 0
	.loc	1 248 21                        ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 239 33                        ; mre.py:239:33
	s_add_i32 s0, s4, 0x4000
	s_and_b32 s0, s0, 0x3f800
	.loc	1 239 23 is_stmt 0              ; mre.py:239:23
	s_lshl_b32 s0, s0, 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, 0
	.loc	1 246 21 is_stmt 1              ; mre.py:246:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 239 33                        ; mre.py:239:33
	s_add_i32 s0, s4, 0xc800
	s_and_b32 s0, s0, 0x3f800
	.loc	1 239 23 is_stmt 0              ; mre.py:239:23
	s_lshl_b32 s0, s0, 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, 0
	.loc	1 246 21 is_stmt 1              ; mre.py:246:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 24                        ; mre.py:240:24
	s_add_u32 s0, s5, 0xa000
	s_addc_u32 s1, s6, 0
	.loc	1 248 21                        ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 24                        ; mre.py:240:24
	s_add_u32 s0, s5, 0xb000
	s_addc_u32 s1, s6, 0
	.loc	1 248 21                        ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 239 33                        ; mre.py:239:33
	s_add_i32 s0, s4, 0x26000
	s_and_b32 s0, s0, 0x3f800
	.loc	1 239 23 is_stmt 0              ; mre.py:239:23
	s_lshl_b32 s0, s0, 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, 0
	.loc	1 246 21 is_stmt 1              ; mre.py:246:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 239 33                        ; mre.py:239:33
	s_add_i32 s4, s4, 0x2e800
	.loc	1 246 21                        ; mre.py:246:21
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 239 33                        ; mre.py:239:33
	s_and_b32 s0, s4, 0x3f800
	.loc	1 239 23 is_stmt 0              ; mre.py:239:23
	s_lshl_b32 s0, s0, 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s18, 0
	.loc	1 246 21 is_stmt 1              ; mre.py:246:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 24                        ; mre.py:240:24
	s_add_u32 s0, s5, 0xe000
	s_addc_u32 s1, s6, 0
	.loc	1 248 21                        ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 240 24                        ; mre.py:240:24
	s_add_u32 s0, s5, 0xf000
	s_addc_u32 s1, s6, 0
	.loc	1 248 21                        ; mre.py:248:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
.Ltmp17:
	.loc	1 104 5                         ; mre.py:104:5 @[ mre.py:250:13 ]
	v_mov_b32_e32 v1, 0
	;;#ASMSTART
	s_nop 0
	;;#ASMEND
.Ltmp18:
.LBB0_14:                               ; %._crit_edge
	.loc	1 253 15                        ; mre.py:253:15
	v_add_u32_e32 v0, 0, v0
	v_add_u32_e32 v1, 0x19000, v0
	.loc	1 254 13                        ; mre.py:254:13
	v_add_u32_e32 v0, 0x18000, v0
	; asyncmark
	; wait_asyncmark(0)
	.loc	1 253 15                        ; mre.py:253:15
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[6:9], v1
	.loc	1 254 13                        ; mre.py:254:13
	ds_read_b128 v[10:13], v0
	.loc	1 253 15                        ; mre.py:253:15
	s_waitcnt lgkmcnt(1)
	v_cvt_f32_f16_e32 v0, v6
	v_cvt_f32_f16_sdwa v1, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	.loc	1 254 13                        ; mre.py:254:13
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_f16_e32 v14, v10
	v_cvt_f32_f16_sdwa v15, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	.loc	1 253 15                        ; mre.py:253:15
	v_cvt_f32_f16_e32 v6, v7
	v_cvt_f32_f16_sdwa v7, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	.loc	1 254 13                        ; mre.py:254:13
	v_cvt_f32_f16_e32 v10, v11
	v_cvt_f32_f16_sdwa v11, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	.loc	1 253 15                        ; mre.py:253:15
	v_cvt_f32_f16_e32 v16, v8
	v_cvt_f32_f16_sdwa v17, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	.loc	1 254 13                        ; mre.py:254:13
	v_cvt_f32_f16_e32 v18, v12
	v_cvt_f32_f16_sdwa v19, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	.loc	1 253 15                        ; mre.py:253:15
	v_cvt_f32_f16_e32 v8, v9
	v_cvt_f32_f16_sdwa v9, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	.loc	1 254 13                        ; mre.py:254:13
	v_cvt_f32_f16_e32 v12, v13
	v_cvt_f32_f16_sdwa v13, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	.loc	1 253 15                        ; mre.py:253:15
	v_pk_add_f32 v[0:1], v[0:1], v[14:15]
	v_pk_add_f32 v[6:7], v[6:7], v[10:11]
	v_pk_add_f32 v[10:11], v[16:17], v[18:19]
	v_pk_add_f32 v[8:9], v[8:9], v[12:13]
	.loc	1 231 5                         ; mre.py:231:5
	v_pk_add_f32 v[0:1], v[0:1], v[6:7]
	v_pk_add_f32 v[6:7], v[10:11], v[8:9]
	s_nop 0
	v_pk_add_f32 v[0:1], v[0:1], v[6:7]
.LBB0_15:
.Ltmp19:
	.file	2 "/home/jbowley/repos/triton/python/triton/language" "standard.py"
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:255:32 ] ]
	s_nop 0
	v_add_f32_e32 v0, v0, v1
.Ltmp20:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:255:25 ]
	s_lshl2_add_u32 s0, s17, 0x18000
.Ltmp21:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:255:32 ] ]
	s_nop 0
	v_add_f32_dpp v0, v0, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp22:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:255:25 ]
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v5, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v1, v5
	v_cmp_eq_u32_e32 vcc, v1, v0
	s_barrier
	s_nop 0
	v_cndmask_b32_e32 v1, v1, v5, vcc
.Ltmp23:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:255:25 ] ]
	v_add_f32_e32 v0, v0, v1
.Ltmp24:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:255:25 ]
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v5, v0
	s_nop 1
	v_permlane16_swap_b32_e32 v1, v5
	v_cmp_eq_u32_e32 vcc, v1, v0
.Ltmp25:
	.loc	1 255 5                         ; mre.py:255:5
	v_or_b32_e32 v2, s17, v2
.Ltmp26:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:255:25 ]
	s_nop 0
	v_cndmask_b32_e32 v1, v1, v5, vcc
.Ltmp27:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:255:25 ] ]
	v_add_f32_e32 v0, v0, v1
.Ltmp28:
	.loc	1 255 5                         ; mre.py:255:5
	v_cmp_eq_u32_e32 vcc, 0, v2
.Ltmp29:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:255:25 ] ]
	s_nop 0
	v_add_f32_dpp v0, v0, v0 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp30:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:255:25 ]
	s_nop 1
	v_mov_b32_dpp v1, v0 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp31:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:255:25 ] ]
	s_nop 1
	v_add_f32_dpp v0, v1, v0 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp32:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:255:25 ]
	v_mov_b32_e32 v1, s0
.Ltmp33:
	.loc	2 353 12 is_stmt 0              ; standard.py:353:12 @[ mre.py:137:12 @[ mre.py:255:63 ] ]
	s_add_i32 s0, s0, 0xfffe8000
.Ltmp34:
	.loc	2 326 12 is_stmt 1              ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:255:25 ] ]
	v_add_f32_dpp v0, v0, v0 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp35:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:255:25 ]
	ds_write_b32 v1, v0
	v_lshlrev_b32_e32 v0, 2, v4
	v_and_b32_e32 v0, 12, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v0, 0x18000, v0
.Ltmp36:
	.loc	1 137 19                        ; mre.py:137:19 @[ mre.py:255:63 ]
	ds_read_i8 v1, v3
.Ltmp37:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:255:25 ]
	ds_read_b32 v4, v0
.Ltmp38:
	.loc	1 138 16                        ; mre.py:138:16 @[ mre.py:255:63 ]
	v_add_u32_e32 v3, 0x10000, v3
.Ltmp39:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:137:12 @[ mre.py:255:63 ] ]
	s_waitcnt lgkmcnt(0)
	s_barrier
.Ltmp40:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:137:12 @[ mre.py:255:63 ] ] ]
	v_add_u32_dpp v1, v1, v1 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp41:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:137:12 @[ mre.py:255:63 ] ]
	v_add_u32_e32 v0, 0xfffe8000, v0
.Ltmp42:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:137:12 @[ mre.py:255:63 ] ] ]
	s_nop 0
	v_add_u32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_u32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_u32_dpp v1, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp43:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:137:12 @[ mre.py:255:63 ] ]
	v_mov_b32_e32 v5, v1
	s_nop 1
	v_mov_b32_dpp v5, v5 row_bcast:15 row_mask:0xa bank_mask:0xf bound_ctrl:1
.Ltmp44:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:137:12 @[ mre.py:255:63 ] ] ]
	v_add_u32_e32 v1, v1, v5
	s_nop 1
	v_add_u32_dpp v1, v1, v1 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp45:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:137:12 @[ mre.py:255:63 ] ]
	s_nop 0
	v_readlane_b32 s1, v1, 63
	v_mov_b32_e32 v1, s0
	s_nop 0
	v_mov_b32_e32 v5, s1
	ds_write_b32 v1, v5
	s_waitcnt lgkmcnt(0)
	s_barrier
.Ltmp46:
	.loc	1 138 16                        ; mre.py:138:16 @[ mre.py:255:63 ]
	ds_read_i8 v3, v3
.Ltmp47:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:137:12 @[ mre.py:255:63 ] ]
	ds_read_b32 v5, v0
.Ltmp48:
	.loc	2 353 12 is_stmt 0              ; standard.py:353:12 @[ mre.py:138:9 @[ mre.py:255:63 ] ]
	s_waitcnt lgkmcnt(0)
	s_barrier
.Ltmp49:
	.loc	2 326 12 is_stmt 1              ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:138:9 @[ mre.py:255:63 ] ] ]
	v_add_u32_dpp v3, v3, v3 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_u32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_u32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_u32_dpp v3, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp50:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:138:9 @[ mre.py:255:63 ] ]
	v_mov_b32_e32 v6, v3
	s_nop 1
	v_mov_b32_dpp v6, v6 row_bcast:15 row_mask:0xa bank_mask:0xf bound_ctrl:1
.Ltmp51:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:138:9 @[ mre.py:255:63 ] ] ]
	v_add_u32_e32 v3, v3, v6
	s_nop 1
	v_add_u32_dpp v3, v3, v3 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp52:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:138:9 @[ mre.py:255:63 ] ]
	s_nop 0
	v_readlane_b32 s0, v3, 63
	s_nop 1
	v_mov_b32_e32 v3, s0
	ds_write_b32 v1, v3
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v6, v0
.Ltmp53:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:255:25 ] ]
	v_add_f32_dpp v0, v4, v4 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp54:
	.loc	2 326 12 is_stmt 0              ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:137:12 @[ mre.py:255:63 ] ] ]
	v_add_u32_dpp v3, v5, v5 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp55:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:138:9 @[ mre.py:255:63 ] ] ]
	s_waitcnt lgkmcnt(0)
	v_add_u32_dpp v5, v6, v6 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp56:
	.loc	2 353 12 is_stmt 1              ; standard.py:353:12 @[ mre.py:255:25 ]
	v_mov_b32_dpp v1, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp57:
	.loc	2 353 12 is_stmt 0              ; standard.py:353:12 @[ mre.py:137:12 @[ mre.py:255:63 ] ]
	v_mov_b32_dpp v4, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp58:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:138:9 @[ mre.py:255:63 ] ]
	v_mov_b32_dpp v6, v5 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp59:
	.loc	1 255 5 is_stmt 1               ; mre.py:255:5
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_17
; %bb.16:
.Ltmp60:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:138:9 @[ mre.py:255:63 ] ] ]
	v_add_u32_e32 v2, v3, v4
.Ltmp61:
	.loc	1 137 12                        ; mre.py:137:12 @[ mre.py:255:63 ]
	v_add3_u32 v2, v2, v6, v5
.Ltmp62:
	.loc	1 255 63                        ; mre.py:255:63
	v_cvt_f32_i32_e32 v2, v2
	.loc	1 255 14 is_stmt 0              ; mre.py:255:14
	s_ashr_i32 s17, s16, 31
	s_lshl_b64 s[0:1], s[16:17], 2
.Ltmp63:
	.loc	2 326 12 is_stmt 1              ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:255:25 ] ]
	v_add_f32_e32 v0, v0, v1
.Ltmp64:
	.loc	1 255 14                        ; mre.py:255:14
	s_add_u32 s0, s10, s0
	.loc	1 255 25 is_stmt 0              ; mre.py:255:25
	v_add_f32_e32 v0, v0, v2
	.loc	1 255 14                        ; mre.py:255:14
	s_addc_u32 s1, s11, s1
	v_mov_b32_e32 v1, 0
	.loc	1 255 5                         ; mre.py:255:5
	global_store_dword v1, v0, s[0:1]
.LBB0_17:
	.loc	1 211 1 is_stmt 1               ; mre.py:211:1
	s_endpgm
.Ltmp65:
.Lfunc_end0:
	.size	mix_kernel, .Lfunc_end0-mix_kernel
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel mix_kernel
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 320
		.amdhsa_user_sgpr_count 16
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 1
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 1
		.amdhsa_user_sgpr_kernarg_preload_length 8
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 20
		.amdhsa_next_free_sgpr 77
		.amdhsa_accum_offset 20
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .Lmix_kernel.num_vgpr, 20
	.set .Lmix_kernel.num_agpr, 0
	.set .Lmix_kernel.numbered_sgpr, 77
	.set .Lmix_kernel.num_named_barrier, 0
	.set .Lmix_kernel.private_seg_size, 0
	.set .Lmix_kernel.uses_vcc, 1
	.set .Lmix_kernel.uses_flat_scratch, 0
	.set .Lmix_kernel.has_dyn_sized_stack, 0
	.set .Lmix_kernel.has_recursion, 0
	.set .Lmix_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4140
; TotalNumSgprs: 83
; NumVgprs: 20
; NumAgprs: 0
; TotalNumVgprs: 20
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 10
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 83
; NumVGPRsForWavesPerEU: 20
; AccumOffset: 20
; Occupancy: 7
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 4
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	14                              ; DW_FORM_strp
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	14                              ; DW_FORM_strp
	.ascii	"\341\177"                      ; DW_AT_APPLE_optimized
	.byte	25                              ; DW_FORM_flag_present
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	32                              ; DW_AT_inline
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	5                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	6                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	7                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	8                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x154 DW_TAG_compile_unit
	.long	.Linfo_string0                  ; DW_AT_producer
	.short	2                               ; DW_AT_language
	.long	.Linfo_string1                  ; DW_AT_name
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.long	.Linfo_string2                  ; DW_AT_comp_dir
                                        ; DW_AT_APPLE_optimized
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.byte	2                               ; Abbrev [2] 0x2a:0x6 DW_TAG_subprogram
	.long	.Linfo_string3                  ; DW_AT_name
	.byte	1                               ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x30:0x12e DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	218                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x4d:0x7b DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp4                          ; DW_AT_low_pc
	.long	.Ltmp10-.Ltmp4                  ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	220                             ; DW_AT_call_line
	.byte	19                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x61:0x29 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp4                          ; DW_AT_low_pc
	.long	.Ltmp5-.Ltmp4                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	111                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x75:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp4                          ; DW_AT_low_pc
	.long	.Ltmp5-.Ltmp4                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	78                              ; DW_AT_call_line
	.byte	19                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x8a:0x29 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp6                          ; DW_AT_low_pc
	.long	.Ltmp7-.Ltmp6                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	112                             ; DW_AT_call_line
	.byte	10                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x9e:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp6                          ; DW_AT_low_pc
	.long	.Ltmp7-.Ltmp6                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	78                              ; DW_AT_call_line
	.byte	19                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0xb3:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp8                          ; DW_AT_low_pc
	.long	.Ltmp9-.Ltmp8                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	117                             ; DW_AT_call_line
	.byte	47                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0xc8:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp11                         ; DW_AT_low_pc
	.long	.Ltmp12-.Ltmp11                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	235                             ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0xdc:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	250                             ; DW_AT_call_line
	.byte	13                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0xe8:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	255                             ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	8                               ; Abbrev [8] 0xf4:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.short	353                             ; DW_AT_call_line
	.byte	12                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x102:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges3                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	255                             ; DW_AT_call_line
	.byte	25                              ; DW_AT_call_column
	.byte	8                               ; Abbrev [8] 0x10e:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges4                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.short	353                             ; DW_AT_call_line
	.byte	12                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x11c:0x41 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges5                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	255                             ; DW_AT_call_line
	.byte	63                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x128:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges6                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	137                             ; DW_AT_call_line
	.byte	12                              ; DW_AT_call_column
	.byte	8                               ; Abbrev [8] 0x134:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges7                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.short	353                             ; DW_AT_call_line
	.byte	12                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x142:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges8                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	138                             ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	8                               ; Abbrev [8] 0x14e:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges9                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.short	353                             ; DW_AT_call_line
	.byte	12                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges3:
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp56-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges4:
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges5:
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp56-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp59-.Lfunc_begin0
	.quad	.Ltmp60-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges6:
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp39-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp55-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp58-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges7:
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp55-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges8:
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp55-.Lfunc_begin0
	.quad	.Ltmp56-.Lfunc_begin0
	.quad	.Ltmp58-.Lfunc_begin0
	.quad	.Ltmp59-.Lfunc_begin0
	.quad	.Ltmp60-.Lfunc_begin0
	.quad	.Ltmp61-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges9:
	.quad	.Ltmp49-.Lfunc_begin0
	.quad	.Ltmp50-.Lfunc_begin0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp52-.Lfunc_begin0
	.quad	.Ltmp55-.Lfunc_begin0
	.quad	.Ltmp56-.Lfunc_begin0
	.quad	.Ltmp60-.Lfunc_begin0
	.quad	.Ltmp61-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0 ; triton
.Linfo_string1:
	.asciz	"mre.py"                        ; string offset=7 ; mre.py
.Linfo_string2:
	.asciz	"/home/jbowley/repos/gfx950-gluon-tutorials/experiments/l2_hbm_latency_mre" ; string offset=14 ; /home/jbowley/repos/gfx950-gluon-tutorials/experiments/l2_hbm_latency_mre
.Linfo_string3:
	.asciz	"mix_kernel"                    ; string offset=88 ; mix_kernel
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
      - .offset:         64
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         68
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         72
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         76
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         78
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         80
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         82
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         84
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         86
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         128
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         144
        .size:           8
        .value_kind:     hidden_hostcall_buffer
      - .offset:         152
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
      - .offset:         160
        .size:           8
        .value_kind:     hidden_heap_v1
      - .offset:         168
        .size:           8
        .value_kind:     hidden_default_queue
      - .offset:         176
        .size:           8
        .value_kind:     hidden_completion_action
      - .offset:         184
        .size:           4
        .value_kind:     hidden_dynamic_lds_size
      - .offset:         264
        .size:           8
        .value_kind:     hidden_queue_ptr
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 320
    .max_flat_workgroup_size: 256
    .name:           mix_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     83
    .sgpr_spill_count: 0
    .symbol:         mix_kernel.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     20
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgpu9.50-amd-amdhsa-unknown-gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
