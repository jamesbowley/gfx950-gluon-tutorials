	.amdgcn_target "amdgpu9.50-amd-amdhsa-unknown-gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	depth_kernel                    ; -- Begin function depth_kernel
	.p2align	8
	.type	depth_kernel,@function
depth_kernel:                           ; @depth_kernel
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.24:
	.file	1 "/home/jbowley/repos/gfx950-gluon-tutorials/experiments/l2_hbm_latency_mre" "mre.py"
	.loc	1 279 0 prologue_end            ; mre.py:279:0
	s_load_dwordx8 s[8:15], s[4:5], 0x0
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.25:
.LBB0_0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	v_and_b32_e32 v4, 0x3ff, v0
.Ltmp1:
	.loc	1 128 5 is_stmt 1               ; mre.py:128:5 @[ mre.py:287:11 ]
	v_and_b32_e32 v2, 63, v0
.Ltmp2:
	.loc	1 336 5                         ; mre.py:336:5
	v_readfirstlane_b32 s0, v4
	s_bfe_u32 s17, s0, 0x20006
.Ltmp3:
	.loc	1 128 5                         ; mre.py:128:5 @[ mre.py:287:11 ]
	v_lshl_or_b32 v5, s17, 6, v2
	v_add_u32_e32 v3, 0, v5
	v_mov_b32_e32 v0, 0
	.loc	1 129 5                         ; mre.py:129:5 @[ mre.py:287:11 ]
	v_add_u32_e32 v1, 0x10000, v3
	.loc	1 128 5                         ; mre.py:128:5 @[ mre.py:287:11 ]
	ds_write_b8 v3, v0
	.loc	1 129 5                         ; mre.py:129:5 @[ mre.py:287:11 ]
	ds_write_b8 v1, v0
.Ltmp4:
	.loc	1 72 12                         ; mre.py:72:12 @[ mre.py:78:19 @[ mre.py:111:11 @[ mre.py:289:19 ] ] ]
	;;#ASMSTART
	s_getreg_b32 s0, hwreg(HW_REG_XCC_ID)
	;;#ASMEND
.Ltmp5:
	.loc	1 111 11                        ; mre.py:111:11 @[ mre.py:289:19 ]
	s_and_b32 s20, s0, 15
.Ltmp6:
	.loc	1 72 12                         ; mre.py:72:12 @[ mre.py:78:19 @[ mre.py:112:10 @[ mre.py:289:19 ] ] ]
	;;#ASMSTART
	s_getreg_b32 s0, hwreg(HW_REG_HW_ID)
	;;#ASMEND
.Ltmp7:
	.loc	1 115 45                        ; mre.py:115:45 @[ mre.py:289:19 ]
	s_bfe_u32 s19, s0, 0x3000d
	.loc	1 115 61 is_stmt 0              ; mre.py:115:61 @[ mre.py:289:19 ]
	s_bfe_u32 s18, s0, 0x40008
	.loc	1 116 20 is_stmt 1              ; mre.py:116:20 @[ mre.py:289:19 ]
	s_lshl_b32 s0, s16, 4
	.loc	1 116 14 is_stmt 0              ; mre.py:116:14 @[ mre.py:289:19 ]
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s6, s12, s0
	.loc	1 116 45                        ; mre.py:116:45 @[ mre.py:289:19 ]
	v_cmp_gt_u32_e32 vcc, 6, v5
	.loc	1 116 14                        ; mre.py:116:14 @[ mre.py:289:19 ]
	s_addc_u32 s7, s13, s1
	.loc	1 116 5                         ; mre.py:116:5 @[ mre.py:289:19 ]
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	.loc	1 115 25 is_stmt 1              ; mre.py:115:25 @[ mre.py:289:19 ]
	v_mov_b32_e32 v7, s18
	v_mov_b32_e32 v8, s19
	.loc	1 115 34 is_stmt 0              ; mre.py:115:34 @[ mre.py:289:19 ]
	v_cmp_eq_u32_e64 s[0:1], 4, v5
	.loc	1 114 18 is_stmt 1              ; mre.py:114:18 @[ mre.py:289:19 ]
	v_subrev_co_u32_e32 v6, vcc, 1, v5
	.loc	1 115 25                        ; mre.py:115:25 @[ mre.py:289:19 ]
	s_nop 0
	v_cndmask_b32_e64 v7, v7, v8, s[0:1]
	.loc	1 114 84                        ; mre.py:114:84 @[ mre.py:289:19 ]
	v_mov_b32_e32 v8, s20
	.loc	1 115 9                         ; mre.py:115:9 @[ mre.py:289:19 ]
	v_cmp_eq_u32_e64 s[0:1], 3, v5
	.loc	1 114 9                         ; mre.py:114:9 @[ mre.py:289:19 ]
	v_mov_b32_e32 v1, s16
	.loc	1 114 84 is_stmt 0              ; mre.py:114:84 @[ mre.py:289:19 ]
	s_nop 0
	v_cndmask_b32_e64 v7, v7, v8, s[0:1]
	.loc	1 114 34                        ; mre.py:114:34 @[ mre.py:289:19 ]
	v_cmp_lt_u32_e64 s[0:1], 1, v6
	s_nop 1
	v_cndmask_b32_e64 v6, 0, v7, s[0:1]
	.loc	1 114 9                         ; mre.py:114:9 @[ mre.py:289:19 ]
	v_cndmask_b32_e32 v1, v6, v1, vcc
	.loc	1 116 14 is_stmt 1              ; mre.py:116:14 @[ mre.py:289:19 ]
	v_lshlrev_b32_e32 v6, 2, v5
	.loc	1 116 5 is_stmt 0               ; mre.py:116:5 @[ mre.py:289:19 ]
	global_store_dword v6, v1, s[6:7]
.LBB0_2:
	.loc	1 0 5                           ; mre.py:0:5
	s_or_b64 exec, exec, s[2:3]
	s_load_dwordx4 s[0:3], s[4:5], 0x24
	.loc	1 117 82 is_stmt 1              ; mre.py:117:82 @[ mre.py:289:19 ]
	v_cmp_eq_u32_e32 vcc, 0, v2
.Ltmp8:
	.loc	1 72 12                         ; mre.py:72:12 @[ mre.py:117:47 @[ mre.py:289:19 ] ]
	;;#ASMSTART
	s_getreg_b32 s21, hwreg(HW_REG_HW_ID)
	;;#ASMEND
.Ltmp9:
	.loc	1 117 5                         ; mre.py:117:5 @[ mre.py:289:19 ]
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_4
; %bb.3:
	.loc	1 0 5 is_stmt 0                 ; mre.py:0:5
	s_lshl_b32 s22, s17, 2
	.loc	1 117 5                         ; mre.py:117:5 @[ mre.py:289:19 ]
	v_mov_b32_e32 v0, s22
	v_mov_b32_e32 v1, s21
	global_store_dword v0, v1, s[6:7] offset:32
.Ltmp10:
.LBB0_4:
	.loc	1 0 5                           ; mre.py:0:5
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u32 s20, s1
	s_cselect_b64 s[6:7], -1, 0
	.loc	1 291 5 is_stmt 1               ; mre.py:291:5
	s_and_b64 s[12:13], s[6:7], exec
	s_cselect_b32 s1, 1, 0
	.loc	1 290 10                        ; mre.py:290:10
	s_cmp_gt_i32 s16, -1
	.loc	1 291 5                         ; mre.py:291:5
	s_cselect_b32 s12, 1, 0
	.loc	1 291 8 is_stmt 0               ; mre.py:291:8
	s_cmp_eq_u32 s0, 1
	.loc	1 291 5                         ; mre.py:291:5
	s_cselect_b32 s20, s1, s12
	.loc	1 293 5 is_stmt 1               ; mre.py:293:5
	s_cmp_eq_u32 s19, s2
	s_cselect_b64 s[12:13], -1, 0
	s_cmp_eq_u32 s18, s3
	s_cselect_b64 s[2:3], -1, 0
	s_and_b64 s[2:3], s[12:13], s[2:3]
	s_and_b64 s[2:3], s[6:7], s[2:3]
	s_and_b64 s[2:3], s[2:3], exec
	s_cselect_b32 s6, 1, 0
	.loc	1 293 8 is_stmt 0               ; mre.py:293:8
	s_cmp_eq_u32 s0, 2
	s_cselect_b64 s[0:1], -1, 0
	.loc	1 293 5                         ; mre.py:293:5
	s_and_b64 s[2:3], s[0:1], exec
	s_cselect_b32 s2, s6, s20
	.loc	1 307 5 is_stmt 1               ; mre.py:307:5
	s_bitcmp0_b32 s2, 0
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v0, 0
	s_cbranch_scc1 .LBB0_21
; %bb.5:                                ; %.preheader
	.loc	1 0 5 is_stmt 0                 ; mre.py:0:5
	s_load_dword s12, s[4:5], 0x20
	s_load_dwordx2 s[6:7], s[4:5], 0x34
	.loc	1 296 25 is_stmt 1              ; mre.py:296:25
	v_lshlrev_b32_e32 v0, 3, v4
	.loc	1 296 13 is_stmt 0              ; mre.py:296:13
	v_lshlrev_b32_e32 v6, 3, v5
	.loc	1 296 25                        ; mre.py:296:25
	v_and_b32_e32 v0, 8, v0
	.loc	1 296 13                        ; mre.py:296:13
	s_movk_i32 s2, 0x7f0
	v_and_or_b32 v0, v6, s2, v0
	.loc	1 303 20 is_stmt 1              ; mre.py:303:20
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s3, s6, 31
	s_mov_b32 s2, s6
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s2, s8, s2
	s_addc_u32 s3, s9, s3
	.loc	1 303 38 is_stmt 0              ; mre.py:303:38
	s_lshl_b32 s4, s16, 11
	.loc	1 293 5 is_stmt 1               ; mre.py:293:5
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s0, 0, s4
	.loc	1 303 38                        ; mre.py:303:38
	s_mul_i32 s0, s0, s7
	.loc	1 303 20 is_stmt 0              ; mre.py:303:20
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s6, s2, s0
	s_addc_u32 s8, s3, s1
	.loc	1 308 9 is_stmt 1               ; mre.py:308:9
	s_cmp_lt_i32 s7, 1
	v_lshlrev_b32_e32 v0, 1, v0
	s_cbranch_scc1 .LBB0_13
; %bb.6:                                ; %.lr.ph
	.loc	1 0 9 is_stmt 0                 ; mre.py:0:9
	s_lshl_b32 s9, s17, 10
	s_add_i32 s9, s9, 0x18000
	.loc	1 308 9                         ; mre.py:308:9
	s_and_b32 s13, s7, 7
	s_mov_b32 s18, 0
	s_cmp_lt_u32 s7, 8
	s_mov_b64 s[0:1], -1
	s_cbranch_scc1 .LBB0_10
; %bb.7:                                ; %.lr.ph.new
	s_and_b32 s18, s7, 0x7ffffff8
	s_mov_b32 s4, 0
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	s_mov_b32 s19, 0
.LBB0_8:                                ; =>This Inner Loop Header: Depth=1
	.loc	1 309 54 is_stmt 1              ; mre.py:309:54
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[0:1], s[4:5], 1
	s_add_u32 s0, s6, s0
	s_addc_u32 s5, s8, s1
	.loc	1 309 13 is_stmt 0              ; mre.py:309:13
	s_and_b32 s1, s5, 0xffff
	s_mov_b32 m0, s9
	.loc	1 309 54                        ; mre.py:309:54
	s_add_u32 s20, s0, 0x1000
	.loc	1 309 13                        ; mre.py:309:13
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	.loc	1 309 54                        ; mre.py:309:54
	s_addc_u32 s1, s5, 0
	.loc	1 309 13                        ; mre.py:309:13
	s_and_b32 s21, s1, 0xffff
	s_mov_b32 s22, s2
	s_mov_b32 s23, s3
	buffer_load_dwordx4 v0, s[20:23], 0 offen lds
	.loc	1 309 54                        ; mre.py:309:54
	s_add_u32 s20, s0, 0x2000
	s_addc_u32 s1, s5, 0
	.loc	1 309 13                        ; mre.py:309:13
	s_and_b32 s21, s1, 0xffff
	buffer_load_dwordx4 v0, s[20:23], 0 offen lds
	.loc	1 309 54                        ; mre.py:309:54
	s_add_u32 s20, s0, 0x3000
	s_addc_u32 s1, s5, 0
	.loc	1 309 13                        ; mre.py:309:13
	s_and_b32 s21, s1, 0xffff
	buffer_load_dwordx4 v0, s[20:23], 0 offen lds
	.loc	1 309 54                        ; mre.py:309:54
	s_add_u32 s20, s0, 0x4000
	s_addc_u32 s1, s5, 0
	.loc	1 309 13                        ; mre.py:309:13
	s_and_b32 s21, s1, 0xffff
	buffer_load_dwordx4 v0, s[20:23], 0 offen lds
	.loc	1 309 54                        ; mre.py:309:54
	s_add_u32 s20, s0, 0x5000
	s_addc_u32 s1, s5, 0
	.loc	1 309 13                        ; mre.py:309:13
	s_and_b32 s21, s1, 0xffff
	buffer_load_dwordx4 v0, s[20:23], 0 offen lds
	.loc	1 309 54                        ; mre.py:309:54
	s_add_u32 s20, s0, 0x6000
	s_addc_u32 s1, s5, 0
	.loc	1 309 13                        ; mre.py:309:13
	s_and_b32 s21, s1, 0xffff
	.loc	1 309 54                        ; mre.py:309:54
	s_add_u32 s0, s0, 0x7000
	s_addc_u32 s1, s5, 0
	.loc	1 309 13                        ; mre.py:309:13
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v0, s[20:23], 0 offen lds
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	.loc	1 308 9 is_stmt 1               ; mre.py:308:9
	s_add_i32 s19, s19, 8
	s_addk_i32 s4, 0x4000
	s_cmp_lg_u32 s18, s19
	s_cbranch_scc1 .LBB0_8
; %bb.9:                                ; %._crit_edge.loopexit.unr-lcssa
	s_cmp_lg_u32 s13, 0
	s_cselect_b64 s[0:1], -1, 0
.LBB0_10:                               ; %Flow10
	.loc	1 0 9 is_stmt 0                 ; mre.py:0:9
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB0_13
; %bb.11:                               ; %.epil.preheader
	.loc	1 308 9 is_stmt 1               ; mre.py:308:9
	s_lshl_b32 s4, s18, 11
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
.LBB0_12:                               ; =>This Inner Loop Header: Depth=1
	.loc	1 309 54                        ; mre.py:309:54
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[0:1], s[4:5], 1
	s_add_u32 s0, s6, s0
	s_addc_u32 s1, s8, s1
	.loc	1 309 13 is_stmt 0              ; mre.py:309:13
	s_mov_b32 m0, s9
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	.loc	1 308 9 is_stmt 1               ; mre.py:308:9
	s_addk_i32 s4, 0x800
	s_add_i32 s13, s13, -1
	s_cmp_lg_u32 s13, 0
	s_cbranch_scc1 .LBB0_12
.LBB0_13:                               ; %._crit_edge
.Ltmp11:
	.loc	1 206 5                         ; mre.py:206:5 @[ mre.py:310:9 ]
	v_mov_b32_e32 v7, 0
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
.Ltmp12:
	.loc	1 311 9                         ; mre.py:311:9
	s_cmp_lt_i32 s12, 1
	s_cbranch_scc1 .LBB0_16
; %bb.14:                               ; %.lr.ph4
	.loc	1 0 9 is_stmt 0                 ; mre.py:0:9
	s_lshl_b32 s5, s17, 10
	s_add_i32 s4, s7, -1
	s_mov_b32 s7, 0
	s_add_i32 s5, s5, 0x19000
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
.LBB0_15:                               ; =>This Inner Loop Header: Depth=1
	.loc	1 314 72 is_stmt 1              ; mre.py:314:72
	s_and_b32 s0, s7, s4
	.loc	1 314 71 is_stmt 0              ; mre.py:314:71
	s_lshl_b32 s0, s0, 11
	.loc	1 314 60                        ; mre.py:314:60
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s6, s0
	s_addc_u32 s1, s8, s1
	.loc	1 314 21                        ; mre.py:314:21
	s_mov_b32 m0, s5
	.loc	1 314 73                        ; mre.py:314:73
	s_add_i32 s9, s7, 1
	.loc	1 314 21                        ; mre.py:314:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 314 72                        ; mre.py:314:72
	s_and_b32 s9, s9, s4
	.loc	1 314 21                        ; mre.py:314:21
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	.loc	1 314 71                        ; mre.py:314:71
	s_lshl_b32 s0, s9, 11
.Ltmp13:
	.loc	1 275 5 is_stmt 1               ; mre.py:275:5 @[ mre.py:319:21 ]
	;;#ASMSTART
	s_waitcnt vmcnt(11)
	;;#ASMEND
.Ltmp14:
	.loc	1 314 60                        ; mre.py:314:60
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s6, s0
	s_addc_u32 s1, s8, s1
	.loc	1 314 73 is_stmt 0              ; mre.py:314:73
	s_add_i32 s9, s7, 2
	.loc	1 314 21                        ; mre.py:314:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 314 72                        ; mre.py:314:72
	s_and_b32 s9, s9, s4
	.loc	1 314 21                        ; mre.py:314:21
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	.loc	1 314 71                        ; mre.py:314:71
	s_lshl_b32 s0, s9, 11
.Ltmp15:
	.loc	1 275 5 is_stmt 1               ; mre.py:275:5 @[ mre.py:319:21 ]
	;;#ASMSTART
	s_waitcnt vmcnt(11)
	;;#ASMEND
.Ltmp16:
	.loc	1 314 60                        ; mre.py:314:60
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s6, s0
	s_addc_u32 s1, s8, s1
	.loc	1 314 73 is_stmt 0              ; mre.py:314:73
	s_add_i32 s9, s7, 3
	.loc	1 314 21                        ; mre.py:314:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 314 72                        ; mre.py:314:72
	s_and_b32 s9, s9, s4
	.loc	1 314 21                        ; mre.py:314:21
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	.loc	1 314 71                        ; mre.py:314:71
	s_lshl_b32 s0, s9, 11
.Ltmp17:
	.loc	1 275 5 is_stmt 1               ; mre.py:275:5 @[ mre.py:319:21 ]
	;;#ASMSTART
	s_waitcnt vmcnt(11)
	;;#ASMEND
.Ltmp18:
	.loc	1 314 60                        ; mre.py:314:60
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s6, s0
	s_addc_u32 s1, s8, s1
	.loc	1 314 73 is_stmt 0              ; mre.py:314:73
	s_add_i32 s9, s7, 4
	.loc	1 314 21                        ; mre.py:314:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 314 72                        ; mre.py:314:72
	s_and_b32 s9, s9, s4
	.loc	1 314 21                        ; mre.py:314:21
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	.loc	1 314 71                        ; mre.py:314:71
	s_lshl_b32 s0, s9, 11
.Ltmp19:
	.loc	1 275 5 is_stmt 1               ; mre.py:275:5 @[ mre.py:319:21 ]
	;;#ASMSTART
	s_waitcnt vmcnt(11)
	;;#ASMEND
.Ltmp20:
	.loc	1 314 60                        ; mre.py:314:60
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s6, s0
	s_addc_u32 s1, s8, s1
	.loc	1 314 73 is_stmt 0              ; mre.py:314:73
	s_add_i32 s9, s7, 5
	.loc	1 314 21                        ; mre.py:314:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 314 72                        ; mre.py:314:72
	s_and_b32 s9, s9, s4
	.loc	1 314 21                        ; mre.py:314:21
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	.loc	1 314 71                        ; mre.py:314:71
	s_lshl_b32 s0, s9, 11
.Ltmp21:
	.loc	1 275 5 is_stmt 1               ; mre.py:275:5 @[ mre.py:319:21 ]
	;;#ASMSTART
	s_waitcnt vmcnt(11)
	;;#ASMEND
.Ltmp22:
	.loc	1 314 60                        ; mre.py:314:60
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s6, s0
	s_addc_u32 s1, s8, s1
	.loc	1 314 73 is_stmt 0              ; mre.py:314:73
	s_add_i32 s9, s7, 6
	.loc	1 314 21                        ; mre.py:314:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 314 72                        ; mre.py:314:72
	s_and_b32 s9, s9, s4
	.loc	1 314 21                        ; mre.py:314:21
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	.loc	1 314 71                        ; mre.py:314:71
	s_lshl_b32 s0, s9, 11
.Ltmp23:
	.loc	1 275 5 is_stmt 1               ; mre.py:275:5 @[ mre.py:319:21 ]
	;;#ASMSTART
	s_waitcnt vmcnt(11)
	;;#ASMEND
.Ltmp24:
	.loc	1 314 60                        ; mre.py:314:60
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s6, s0
	s_addc_u32 s1, s8, s1
	.loc	1 314 73 is_stmt 0              ; mre.py:314:73
	s_add_i32 s9, s7, 7
	.loc	1 314 21                        ; mre.py:314:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 314 72                        ; mre.py:314:72
	s_and_b32 s9, s9, s4
	.loc	1 314 21                        ; mre.py:314:21
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	.loc	1 314 71                        ; mre.py:314:71
	s_lshl_b32 s0, s9, 11
.Ltmp25:
	.loc	1 275 5 is_stmt 1               ; mre.py:275:5 @[ mre.py:319:21 ]
	;;#ASMSTART
	s_waitcnt vmcnt(11)
	;;#ASMEND
.Ltmp26:
	.loc	1 314 60                        ; mre.py:314:60
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s6, s0
	s_addc_u32 s1, s8, s1
	.loc	1 314 73 is_stmt 0              ; mre.py:314:73
	s_add_i32 s9, s7, 8
	.loc	1 314 21                        ; mre.py:314:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 314 72                        ; mre.py:314:72
	s_and_b32 s9, s9, s4
	.loc	1 314 21                        ; mre.py:314:21
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	.loc	1 314 71                        ; mre.py:314:71
	s_lshl_b32 s0, s9, 11
.Ltmp27:
	.loc	1 275 5 is_stmt 1               ; mre.py:275:5 @[ mre.py:319:21 ]
	;;#ASMSTART
	s_waitcnt vmcnt(11)
	;;#ASMEND
.Ltmp28:
	.loc	1 314 60                        ; mre.py:314:60
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s6, s0
	s_addc_u32 s1, s8, s1
	.loc	1 314 73 is_stmt 0              ; mre.py:314:73
	s_add_i32 s9, s7, 9
	.loc	1 314 21                        ; mre.py:314:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 314 72                        ; mre.py:314:72
	s_and_b32 s9, s9, s4
	.loc	1 314 21                        ; mre.py:314:21
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	.loc	1 314 71                        ; mre.py:314:71
	s_lshl_b32 s0, s9, 11
.Ltmp29:
	.loc	1 275 5 is_stmt 1               ; mre.py:275:5 @[ mre.py:319:21 ]
	;;#ASMSTART
	s_waitcnt vmcnt(11)
	;;#ASMEND
.Ltmp30:
	.loc	1 314 60                        ; mre.py:314:60
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s6, s0
	s_addc_u32 s1, s8, s1
	.loc	1 314 73 is_stmt 0              ; mre.py:314:73
	s_add_i32 s9, s7, 10
	.loc	1 314 21                        ; mre.py:314:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 314 72                        ; mre.py:314:72
	s_and_b32 s9, s9, s4
	.loc	1 314 21                        ; mre.py:314:21
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	.loc	1 314 71                        ; mre.py:314:71
	s_lshl_b32 s0, s9, 11
.Ltmp31:
	.loc	1 275 5 is_stmt 1               ; mre.py:275:5 @[ mre.py:319:21 ]
	;;#ASMSTART
	s_waitcnt vmcnt(11)
	;;#ASMEND
.Ltmp32:
	.loc	1 314 60                        ; mre.py:314:60
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s6, s0
	s_addc_u32 s1, s8, s1
	.loc	1 314 73 is_stmt 0              ; mre.py:314:73
	s_add_i32 s9, s7, 11
	.loc	1 314 21                        ; mre.py:314:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 314 72                        ; mre.py:314:72
	s_and_b32 s9, s9, s4
	.loc	1 314 21                        ; mre.py:314:21
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	.loc	1 314 71                        ; mre.py:314:71
	s_lshl_b32 s0, s9, 11
.Ltmp33:
	.loc	1 275 5 is_stmt 1               ; mre.py:275:5 @[ mre.py:319:21 ]
	;;#ASMSTART
	s_waitcnt vmcnt(11)
	;;#ASMEND
.Ltmp34:
	.loc	1 314 60                        ; mre.py:314:60
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s6, s0
	s_addc_u32 s1, s8, s1
	.loc	1 314 73 is_stmt 0              ; mre.py:314:73
	s_add_i32 s9, s7, 12
	.loc	1 314 21                        ; mre.py:314:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 314 72                        ; mre.py:314:72
	s_and_b32 s9, s9, s4
	.loc	1 314 21                        ; mre.py:314:21
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	.loc	1 314 71                        ; mre.py:314:71
	s_lshl_b32 s0, s9, 11
.Ltmp35:
	.loc	1 275 5 is_stmt 1               ; mre.py:275:5 @[ mre.py:319:21 ]
	;;#ASMSTART
	s_waitcnt vmcnt(11)
	;;#ASMEND
.Ltmp36:
	.loc	1 314 60                        ; mre.py:314:60
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s6, s0
	s_addc_u32 s1, s8, s1
	.loc	1 314 73 is_stmt 0              ; mre.py:314:73
	s_add_i32 s9, s7, 13
	.loc	1 314 21                        ; mre.py:314:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 314 72                        ; mre.py:314:72
	s_and_b32 s9, s9, s4
	.loc	1 314 21                        ; mre.py:314:21
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	.loc	1 314 71                        ; mre.py:314:71
	s_lshl_b32 s0, s9, 11
.Ltmp37:
	.loc	1 275 5 is_stmt 1               ; mre.py:275:5 @[ mre.py:319:21 ]
	;;#ASMSTART
	s_waitcnt vmcnt(11)
	;;#ASMEND
.Ltmp38:
	.loc	1 314 60                        ; mre.py:314:60
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s6, s0
	s_addc_u32 s1, s8, s1
	.loc	1 314 73 is_stmt 0              ; mre.py:314:73
	s_add_i32 s9, s7, 14
	.loc	1 314 21                        ; mre.py:314:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 314 72                        ; mre.py:314:72
	s_and_b32 s9, s9, s4
	.loc	1 314 21                        ; mre.py:314:21
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	.loc	1 314 71                        ; mre.py:314:71
	s_lshl_b32 s0, s9, 11
.Ltmp39:
	.loc	1 275 5 is_stmt 1               ; mre.py:275:5 @[ mre.py:319:21 ]
	;;#ASMSTART
	s_waitcnt vmcnt(11)
	;;#ASMEND
.Ltmp40:
	.loc	1 314 60                        ; mre.py:314:60
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s6, s0
	s_addc_u32 s1, s8, s1
	.loc	1 314 73 is_stmt 0              ; mre.py:314:73
	s_add_i32 s9, s7, 15
	.loc	1 314 21                        ; mre.py:314:21
	s_and_b32 s1, s1, 0xffff
	.loc	1 314 72                        ; mre.py:314:72
	s_and_b32 s9, s9, s4
	.loc	1 314 21                        ; mre.py:314:21
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	.loc	1 314 71                        ; mre.py:314:71
	s_lshl_b32 s0, s9, 11
.Ltmp41:
	.loc	1 275 5 is_stmt 1               ; mre.py:275:5 @[ mre.py:319:21 ]
	;;#ASMSTART
	s_waitcnt vmcnt(11)
	;;#ASMEND
.Ltmp42:
	.loc	1 314 60                        ; mre.py:314:60
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s6, s0
	s_addc_u32 s1, s8, s1
	.loc	1 314 21 is_stmt 0              ; mre.py:314:21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
.Ltmp43:
	.loc	1 275 5 is_stmt 1               ; mre.py:275:5 @[ mre.py:319:21 ]
	;;#ASMSTART
	s_waitcnt vmcnt(11)
	;;#ASMEND
.Ltmp44:
	.loc	1 320 19                        ; mre.py:320:19
	s_add_i32 s0, s7, 16
.Ltmp45:
	.loc	1 104 5                         ; mre.py:104:5 @[ mre.py:321:13 ]
	;;#ASMSTART
	s_nop 0
	;;#ASMEND
.Ltmp46:
	.loc	1 311 9                         ; mre.py:311:9
	s_add_i32 s12, s12, -1
	.loc	1 320 18                        ; mre.py:320:18
	s_and_b32 s7, s0, s4
	.loc	1 311 9                         ; mre.py:311:9
	s_cmp_lg_u32 s12, 0
	s_cbranch_scc1 .LBB0_15
.LBB0_16:                               ; %._crit_edge5
	; asyncmark
	; wait_asyncmark(0)
.Ltmp47:
	.loc	1 89 19                         ; mre.py:89:19 @[ mre.py:325:14 ]
	s_waitcnt vmcnt(0)
	s_barrier
	;;#ASMSTART
	s_memtime s[0:1]
s_waitcnt lgkmcnt(0)
	;;#ASMEND
.Ltmp48:
	.loc	1 83 19                         ; mre.py:83:19 @[ mre.py:326:14 ]
	;;#ASMSTART
	s_memrealtime s[4:5]
s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_add_u32 s2, s4, 0x7d0
	s_addc_u32 s3, s5, 0
	v_mov_b64_e32 v[0:1], s[2:3]
.Ltmp49:
	.loc	1 0 19 is_stmt 0                ; :0:19
.Ltmp50:
	.p2align	5, , 4
.LBB0_17:                               ; =>This Inner Loop Header: Depth=1
	.loc	1 83 19 is_stmt 1               ; mre.py:83:19 @[ mre.py:329:27 ]
	;;#ASMSTART
	s_memrealtime s[2:3]
s_waitcnt lgkmcnt(0)
	;;#ASMEND
.Ltmp51:
	.loc	1 329 27                        ; mre.py:329:27
	s_nop 0
	v_cmp_lt_i64_e32 vcc, s[2:3], v[0:1]
	.loc	1 328 15                        ; mre.py:328:15
	s_cbranch_vccnz .LBB0_17
; %bb.18:
.Ltmp52:
	.loc	1 89 19                         ; mre.py:89:19 @[ mre.py:330:14 ]
	v_mov_b32_e32 v0, 0
	;;#ASMSTART
	s_memtime s[6:7]
s_waitcnt lgkmcnt(0)
	;;#ASMEND
.Ltmp53:
	.loc	1 83 19                         ; mre.py:83:19 @[ mre.py:331:14 ]
	;;#ASMSTART
	s_memrealtime s[8:9]
s_waitcnt lgkmcnt(0)
	;;#ASMEND
.Ltmp54:
	.loc	1 333 112                       ; mre.py:333:112
	v_cmp_gt_u32_e32 vcc, 4, v5
	.loc	1 333 9 is_stmt 0               ; mre.py:333:9
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_20
; %bb.19:
	.loc	1 333 79                        ; mre.py:333:79
	v_mov_b32_e32 v0, s9
	v_mov_b32_e32 v1, s7
	.loc	1 333 88                        ; mre.py:333:88
	v_cmp_eq_u32_e32 vcc, 2, v5
	.loc	1 333 79                        ; mre.py:333:79
	v_mov_b32_e32 v7, s6
	s_nop 0
	v_cndmask_b32_e32 v0, v0, v1, vcc
	v_mov_b32_e32 v1, s8
	v_cndmask_b32_e32 v1, v1, v7, vcc
	.loc	1 333 58                        ; mre.py:333:58
	v_mov_b32_e32 v7, s4
	.loc	1 333 67                        ; mre.py:333:67
	v_cmp_eq_u32_e32 vcc, 1, v5
	.loc	1 333 58                        ; mre.py:333:58
	s_nop 1
	v_cndmask_b32_e32 v7, v1, v7, vcc
	v_mov_b32_e32 v1, s5
	v_cndmask_b32_e32 v0, v0, v1, vcc
	.loc	1 333 37                        ; mre.py:333:37
	v_mov_b32_e32 v1, s1
	.loc	1 333 46                        ; mre.py:333:46
	v_cmp_eq_u32_e32 vcc, 0, v5
	.loc	1 333 37                        ; mre.py:333:37
	s_nop 1
	v_cndmask_b32_e32 v1, v0, v1, vcc
	v_mov_b32_e32 v0, s0
	.loc	1 333 24                        ; mre.py:333:24
	s_lshl_b32 s0, s16, 2
	.loc	1 333 18                        ; mre.py:333:18
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 3
	s_add_u32 s0, s14, s0
	.loc	1 333 37                        ; mre.py:333:37
	v_cndmask_b32_e32 v0, v7, v0, vcc
	.loc	1 333 18                        ; mre.py:333:18
	s_addc_u32 s1, s15, s1
	.loc	1 333 9                         ; mre.py:333:9
	global_store_dwordx2 v6, v[0:1], s[0:1]
.LBB0_20:
	.loc	1 0 9                           ; mre.py:0:9
	s_or_b64 exec, exec, s[2:3]
	.loc	1 334 15 is_stmt 1              ; mre.py:334:15
	v_lshl_add_u32 v0, v5, 4, 0
	v_add_u32_e32 v1, 0x19000, v0
	.loc	1 335 13                        ; mre.py:335:13
	v_add_u32_e32 v0, 0x18000, v0
	.loc	1 334 15                        ; mre.py:334:15
	ds_read_b128 v[6:9], v1
	.loc	1 335 13                        ; mre.py:335:13
	ds_read_b128 v[10:13], v0
	.loc	1 334 15                        ; mre.py:334:15
	s_waitcnt lgkmcnt(1)
	v_cvt_f32_f16_e32 v0, v6
	v_cvt_f32_f16_sdwa v1, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	.loc	1 335 13                        ; mre.py:335:13
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_f16_e32 v14, v10
	v_cvt_f32_f16_sdwa v15, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	.loc	1 334 15                        ; mre.py:334:15
	v_cvt_f32_f16_e32 v6, v7
	v_cvt_f32_f16_sdwa v7, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	.loc	1 335 13                        ; mre.py:335:13
	v_cvt_f32_f16_e32 v10, v11
	v_cvt_f32_f16_sdwa v11, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	.loc	1 334 15                        ; mre.py:334:15
	v_cvt_f32_f16_e32 v16, v8
	v_cvt_f32_f16_sdwa v17, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	.loc	1 335 13                        ; mre.py:335:13
	v_cvt_f32_f16_e32 v18, v12
	v_cvt_f32_f16_sdwa v19, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	.loc	1 334 15                        ; mre.py:334:15
	v_cvt_f32_f16_e32 v8, v9
	v_cvt_f32_f16_sdwa v9, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	.loc	1 335 13                        ; mre.py:335:13
	v_cvt_f32_f16_e32 v12, v13
	v_cvt_f32_f16_sdwa v13, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	.loc	1 334 15                        ; mre.py:334:15
	v_pk_add_f32 v[0:1], v[0:1], v[14:15]
	v_pk_add_f32 v[6:7], v[6:7], v[10:11]
	v_pk_add_f32 v[10:11], v[16:17], v[18:19]
	v_pk_add_f32 v[8:9], v[8:9], v[12:13]
	.loc	1 307 5                         ; mre.py:307:5
	v_pk_add_f32 v[0:1], v[0:1], v[6:7]
	v_pk_add_f32 v[6:7], v[10:11], v[8:9]
	s_nop 0
	v_pk_add_f32 v[0:1], v[0:1], v[6:7]
.LBB0_21:
.Ltmp55:
	.file	2 "/home/jbowley/repos/triton/python/triton/language" "standard.py"
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:336:32 ] ]
	s_nop 0
	v_add_f32_e32 v0, v0, v1
.Ltmp56:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:336:25 ]
	s_lshl2_add_u32 s0, s17, 0x18000
.Ltmp57:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:336:32 ] ]
	s_nop 0
	v_add_f32_dpp v0, v0, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp58:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:336:25 ]
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v5, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v1, v5
	v_cmp_eq_u32_e32 vcc, v1, v0
	s_barrier
	s_nop 0
	v_cndmask_b32_e32 v1, v1, v5, vcc
.Ltmp59:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:336:25 ] ]
	v_add_f32_e32 v0, v0, v1
.Ltmp60:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:336:25 ]
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v5, v0
	s_nop 1
	v_permlane16_swap_b32_e32 v1, v5
	v_cmp_eq_u32_e32 vcc, v1, v0
.Ltmp61:
	.loc	1 336 5                         ; mre.py:336:5
	v_or_b32_e32 v2, s17, v2
.Ltmp62:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:336:25 ]
	s_nop 0
	v_cndmask_b32_e32 v1, v1, v5, vcc
.Ltmp63:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:336:25 ] ]
	v_add_f32_e32 v0, v0, v1
.Ltmp64:
	.loc	1 336 5                         ; mre.py:336:5
	v_cmp_eq_u32_e32 vcc, 0, v2
.Ltmp65:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:336:25 ] ]
	s_nop 0
	v_add_f32_dpp v0, v0, v0 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp66:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:336:25 ]
	s_nop 1
	v_mov_b32_dpp v1, v0 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp67:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:336:25 ] ]
	s_nop 1
	v_add_f32_dpp v0, v1, v0 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp68:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:336:25 ]
	v_mov_b32_e32 v1, s0
.Ltmp69:
	.loc	2 353 12 is_stmt 0              ; standard.py:353:12 @[ mre.py:137:12 @[ mre.py:336:63 ] ]
	s_add_i32 s0, s0, 0xfffe8000
.Ltmp70:
	.loc	2 326 12 is_stmt 1              ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:336:25 ] ]
	v_add_f32_dpp v0, v0, v0 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp71:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:336:25 ]
	ds_write_b32 v1, v0
	v_lshlrev_b32_e32 v0, 2, v4
	v_and_b32_e32 v0, 12, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v0, 0x18000, v0
.Ltmp72:
	.loc	1 137 19                        ; mre.py:137:19 @[ mre.py:336:63 ]
	ds_read_i8 v1, v3
.Ltmp73:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:336:25 ]
	ds_read_b32 v4, v0
.Ltmp74:
	.loc	1 138 16                        ; mre.py:138:16 @[ mre.py:336:63 ]
	v_add_u32_e32 v3, 0x10000, v3
.Ltmp75:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:137:12 @[ mre.py:336:63 ] ]
	s_waitcnt lgkmcnt(0)
	s_barrier
.Ltmp76:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:137:12 @[ mre.py:336:63 ] ] ]
	v_add_u32_dpp v1, v1, v1 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp77:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:137:12 @[ mre.py:336:63 ] ]
	v_add_u32_e32 v0, 0xfffe8000, v0
.Ltmp78:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:137:12 @[ mre.py:336:63 ] ] ]
	s_nop 0
	v_add_u32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_u32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_u32_dpp v1, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp79:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:137:12 @[ mre.py:336:63 ] ]
	v_mov_b32_e32 v5, v1
	s_nop 1
	v_mov_b32_dpp v5, v5 row_bcast:15 row_mask:0xa bank_mask:0xf bound_ctrl:1
.Ltmp80:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:137:12 @[ mre.py:336:63 ] ] ]
	v_add_u32_e32 v1, v1, v5
	s_nop 1
	v_add_u32_dpp v1, v1, v1 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp81:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:137:12 @[ mre.py:336:63 ] ]
	s_nop 0
	v_readlane_b32 s1, v1, 63
	v_mov_b32_e32 v1, s0
	s_nop 0
	v_mov_b32_e32 v5, s1
	ds_write_b32 v1, v5
	s_waitcnt lgkmcnt(0)
	s_barrier
.Ltmp82:
	.loc	1 138 16                        ; mre.py:138:16 @[ mre.py:336:63 ]
	ds_read_i8 v3, v3
.Ltmp83:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:137:12 @[ mre.py:336:63 ] ]
	ds_read_b32 v5, v0
.Ltmp84:
	.loc	2 353 12 is_stmt 0              ; standard.py:353:12 @[ mre.py:138:9 @[ mre.py:336:63 ] ]
	s_waitcnt lgkmcnt(0)
	s_barrier
.Ltmp85:
	.loc	2 326 12 is_stmt 1              ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:138:9 @[ mre.py:336:63 ] ] ]
	v_add_u32_dpp v3, v3, v3 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_u32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_u32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_u32_dpp v3, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp86:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:138:9 @[ mre.py:336:63 ] ]
	v_mov_b32_e32 v6, v3
	s_nop 1
	v_mov_b32_dpp v6, v6 row_bcast:15 row_mask:0xa bank_mask:0xf bound_ctrl:1
.Ltmp87:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:138:9 @[ mre.py:336:63 ] ] ]
	v_add_u32_e32 v3, v3, v6
	s_nop 1
	v_add_u32_dpp v3, v3, v3 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp88:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:138:9 @[ mre.py:336:63 ] ]
	s_nop 0
	v_readlane_b32 s0, v3, 63
	s_nop 1
	v_mov_b32_e32 v3, s0
	ds_write_b32 v1, v3
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v6, v0
.Ltmp89:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:336:25 ] ]
	v_add_f32_dpp v0, v4, v4 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp90:
	.loc	2 326 12 is_stmt 0              ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:137:12 @[ mre.py:336:63 ] ] ]
	v_add_u32_dpp v3, v5, v5 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp91:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:138:9 @[ mre.py:336:63 ] ] ]
	s_waitcnt lgkmcnt(0)
	v_add_u32_dpp v5, v6, v6 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp92:
	.loc	2 353 12 is_stmt 1              ; standard.py:353:12 @[ mre.py:336:25 ]
	v_mov_b32_dpp v1, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp93:
	.loc	2 353 12 is_stmt 0              ; standard.py:353:12 @[ mre.py:137:12 @[ mre.py:336:63 ] ]
	v_mov_b32_dpp v4, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp94:
	.loc	2 353 12                        ; standard.py:353:12 @[ mre.py:138:9 @[ mre.py:336:63 ] ]
	v_mov_b32_dpp v6, v5 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
.Ltmp95:
	.loc	1 336 5 is_stmt 1               ; mre.py:336:5
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_23
; %bb.22:
.Ltmp96:
	.loc	2 326 12                        ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:138:9 @[ mre.py:336:63 ] ] ]
	v_add_u32_e32 v2, v3, v4
.Ltmp97:
	.loc	1 137 12                        ; mre.py:137:12 @[ mre.py:336:63 ]
	v_add3_u32 v2, v2, v6, v5
.Ltmp98:
	.loc	1 336 63                        ; mre.py:336:63
	v_cvt_f32_i32_e32 v2, v2
	.loc	1 336 14 is_stmt 0              ; mre.py:336:14
	s_ashr_i32 s17, s16, 31
	s_lshl_b64 s[0:1], s[16:17], 2
.Ltmp99:
	.loc	2 326 12 is_stmt 1              ; standard.py:326:12 @[ standard.py:353:12 @[ mre.py:336:25 ] ]
	v_add_f32_e32 v0, v0, v1
.Ltmp100:
	.loc	1 336 14                        ; mre.py:336:14
	s_add_u32 s0, s10, s0
	.loc	1 336 25 is_stmt 0              ; mre.py:336:25
	v_add_f32_e32 v0, v0, v2
	.loc	1 336 14                        ; mre.py:336:14
	s_addc_u32 s1, s11, s1
	v_mov_b32_e32 v1, 0
	.loc	1 336 5                         ; mre.py:336:5
	global_store_dword v1, v0, s[0:1]
.LBB0_23:
	.loc	1 279 1 is_stmt 1               ; mre.py:279:1
	s_endpgm
.Ltmp101:
.Lfunc_end0:
	.size	depth_kernel, .Lfunc_end0-depth_kernel
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel depth_kernel
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 344
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
		.amdhsa_next_free_sgpr 24
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
	.set .Ldepth_kernel.num_vgpr, 20
	.set .Ldepth_kernel.num_agpr, 0
	.set .Ldepth_kernel.numbered_sgpr, 24
	.set .Ldepth_kernel.num_named_barrier, 0
	.set .Ldepth_kernel.private_seg_size, 0
	.set .Ldepth_kernel.uses_vcc, 1
	.set .Ldepth_kernel.uses_flat_scratch, 0
	.set .Ldepth_kernel.has_dyn_sized_stack, 0
	.set .Ldepth_kernel.has_recursion, 0
	.set .Ldepth_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3384
; TotalNumSgprs: 30
; NumVgprs: 20
; NumAgprs: 0
; TotalNumVgprs: 20
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 30
; NumVGPRsForWavesPerEU: 20
; AccumOffset: 20
; Occupancy: 8
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
	.byte	5                               ; DW_FORM_data2
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
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	6                               ; Abbreviation Code
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
	.byte	7                               ; Abbreviation Code
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
	.byte	8                               ; Abbreviation Code
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
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	9                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
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
	.byte	10                              ; Abbreviation Code
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
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x1d9 DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x1b3 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	287                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x4e:0x7c DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp4                          ; DW_AT_low_pc
	.long	.Ltmp10-.Ltmp4                  ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	289                             ; DW_AT_call_line
	.byte	19                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x63:0x29 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp4                          ; DW_AT_low_pc
	.long	.Ltmp5-.Ltmp4                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	111                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x77:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp4                          ; DW_AT_low_pc
	.long	.Ltmp5-.Ltmp4                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	78                              ; DW_AT_call_line
	.byte	19                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x8c:0x29 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp6                          ; DW_AT_low_pc
	.long	.Ltmp7-.Ltmp6                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	112                             ; DW_AT_call_line
	.byte	10                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0xa0:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp6                          ; DW_AT_low_pc
	.long	.Ltmp7-.Ltmp6                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	78                              ; DW_AT_call_line
	.byte	19                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0xb5:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp8                          ; DW_AT_low_pc
	.long	.Ltmp9-.Ltmp8                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	117                             ; DW_AT_call_line
	.byte	47                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	8                               ; Abbrev [8] 0xca:0x15 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp11                         ; DW_AT_low_pc
	.long	.Ltmp12-.Ltmp11                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	310                             ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0xdf:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	319                             ; DW_AT_call_line
	.byte	21                              ; DW_AT_call_column
	.byte	8                               ; Abbrev [8] 0xec:0x15 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp45                         ; DW_AT_low_pc
	.long	.Ltmp46-.Ltmp45                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	321                             ; DW_AT_call_line
	.byte	13                              ; DW_AT_call_column
	.byte	8                               ; Abbrev [8] 0x101:0x15 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp47                         ; DW_AT_low_pc
	.long	.Ltmp48-.Ltmp47                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	325                             ; DW_AT_call_line
	.byte	14                              ; DW_AT_call_column
	.byte	8                               ; Abbrev [8] 0x116:0x15 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp48                         ; DW_AT_low_pc
	.long	.Ltmp49-.Ltmp48                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	326                             ; DW_AT_call_line
	.byte	14                              ; DW_AT_call_column
	.byte	8                               ; Abbrev [8] 0x12b:0x15 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp49                         ; DW_AT_low_pc
	.long	.Ltmp51-.Ltmp49                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	329                             ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	8                               ; Abbrev [8] 0x140:0x15 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp52                         ; DW_AT_low_pc
	.long	.Ltmp53-.Ltmp52                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	330                             ; DW_AT_call_line
	.byte	14                              ; DW_AT_call_column
	.byte	8                               ; Abbrev [8] 0x155:0x15 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp53                         ; DW_AT_low_pc
	.long	.Ltmp54-.Ltmp53                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	331                             ; DW_AT_call_line
	.byte	14                              ; DW_AT_call_column
	.byte	9                               ; Abbrev [9] 0x16a:0x1b DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	336                             ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x177:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.short	353                             ; DW_AT_call_line
	.byte	12                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	9                               ; Abbrev [9] 0x185:0x1b DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges3                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	336                             ; DW_AT_call_line
	.byte	25                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x192:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges4                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.short	353                             ; DW_AT_call_line
	.byte	12                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	9                               ; Abbrev [9] 0x1a0:0x42 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges5                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	336                             ; DW_AT_call_line
	.byte	63                              ; DW_AT_call_column
	.byte	10                              ; Abbrev [10] 0x1ad:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges6                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	137                             ; DW_AT_call_line
	.byte	12                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x1b9:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges7                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.short	353                             ; DW_AT_call_line
	.byte	12                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	10                              ; Abbrev [10] 0x1c7:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges8                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	138                             ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x1d3:0xd DW_TAG_inlined_subroutine
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
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	.Ltmp39-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
	.quad	.Ltmp55-.Lfunc_begin0
	.quad	.Ltmp56-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp58-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges3:
	.quad	.Ltmp56-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp58-.Lfunc_begin0
	.quad	.Ltmp61-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp92-.Lfunc_begin0
	.quad	.Ltmp93-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	.Ltmp100-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges4:
	.quad	.Ltmp59-.Lfunc_begin0
	.quad	.Ltmp60-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	.Ltmp100-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges5:
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp92-.Lfunc_begin0
	.quad	.Ltmp93-.Lfunc_begin0
	.quad	.Ltmp95-.Lfunc_begin0
	.quad	.Ltmp96-.Lfunc_begin0
	.quad	.Ltmp98-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges6:
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	.Ltmp84-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp91-.Lfunc_begin0
	.quad	.Ltmp93-.Lfunc_begin0
	.quad	.Ltmp94-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges7:
	.quad	.Ltmp76-.Lfunc_begin0
	.quad	.Ltmp77-.Lfunc_begin0
	.quad	.Ltmp78-.Lfunc_begin0
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	.Ltmp80-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp91-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges8:
	.quad	.Ltmp84-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp91-.Lfunc_begin0
	.quad	.Ltmp92-.Lfunc_begin0
	.quad	.Ltmp94-.Lfunc_begin0
	.quad	.Ltmp95-.Lfunc_begin0
	.quad	.Ltmp96-.Lfunc_begin0
	.quad	.Ltmp97-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges9:
	.quad	.Ltmp85-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp87-.Lfunc_begin0
	.quad	.Ltmp88-.Lfunc_begin0
	.quad	.Ltmp91-.Lfunc_begin0
	.quad	.Ltmp92-.Lfunc_begin0
	.quad	.Ltmp96-.Lfunc_begin0
	.quad	.Ltmp97-.Lfunc_begin0
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
	.asciz	"depth_kernel"                  ; string offset=88 ; depth_kernel
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
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
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
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     global_buffer
      - .offset:         88
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         92
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         96
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         100
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         102
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         104
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         106
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         108
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         110
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         128
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         136
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         144
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         152
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         168
        .size:           8
        .value_kind:     hidden_hostcall_buffer
      - .offset:         176
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
      - .offset:         184
        .size:           8
        .value_kind:     hidden_heap_v1
      - .offset:         192
        .size:           8
        .value_kind:     hidden_default_queue
      - .offset:         200
        .size:           8
        .value_kind:     hidden_completion_action
      - .offset:         208
        .size:           4
        .value_kind:     hidden_dynamic_lds_size
      - .offset:         288
        .size:           8
        .value_kind:     hidden_queue_ptr
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 344
    .max_flat_workgroup_size: 256
    .name:           depth_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     30
    .sgpr_spill_count: 0
    .symbol:         depth_kernel.kd
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
