#!/bin/bash
# ROCm 7.14.0 (TheRock, hipBLASLt 1.4.1) from /opt/venv, clean environment, GPU 3 only (hipBLASLt issue runs).
#   run_rocm714_gpu3.sh <command> ...
V=/opt/venv
S=$V/lib/python3.12/site-packages
D=$S/_rocm_sdk_devel
exec env -i HOME="$HOME" USER="${USER:-root}" TERM="${TERM:-xterm}" \
  HIP_VISIBLE_DEVICES=3 HVR_ENV=rocm714 TMPDIR=/home/jbowley/hvr_tmp/rocm714 \
  ROCM_PATH=$D ROCM_HOME=$D ROCM_BIN=$D/bin \
  HIP_DEVICE_LIB_PATH=$D/lib/llvm/amdgcn/bitcode ROCM_DEVICE_LIB_PATH=$D/lib/llvm/amdgcn/bitcode \
  LD_LIBRARY_PATH=$D/lib/host-math/lib:$D/lib/rocm_sysdeps/lib \
  LIBRARY_PATH=$D/lib/host-math/lib:$D/lib/rocm_sysdeps/lib \
  C_INCLUDE_PATH=$D/lib/rocm_sysdeps/include CPLUS_INCLUDE_PATH=$D/lib/rocm_sysdeps/include \
  PATH=$D/bin:$V/bin:/usr/local/bin:/usr/bin:/bin \
  "$@"
