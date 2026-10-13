#!/bin/bash
# TheRock nightly (torch 2.15.0a0+rocm10.2.0a20260929) with the llirSched Triton and AITER.
# Clean environment: the container exports ROCm 7.14 paths from /opt/venv. GPU 3 only (hipBLASLt issue runs).
#   run_nightly.sh python script.py ...
V=/home/jbowley/venvs/rocm-nightly-20260929
D=$V/lib/python3.12/site-packages/_rocm_sdk_devel
X=/home/jbowley/repos/gfx950-gluon-tutorials/experiments/hip_version_repro
exec env -i HOME="$HOME" USER="${USER:-root}" TERM="${TERM:-xterm}" \
  HIP_VISIBLE_DEVICES=3 HVR_ENV=nightly TMPDIR=/home/jbowley/hvr_tmp/nightly \
  ROCM_PATH=$D ROCM_HOME=$D ROCM_BIN=$D/bin \
  HIP_DEVICE_LIB_PATH=$D/lib/llvm/amdgcn/bitcode ROCM_DEVICE_LIB_PATH=$D/lib/llvm/amdgcn/bitcode \
  LD_LIBRARY_PATH=$D/lib/host-math/lib:$D/lib/rocm_sysdeps/lib \
  LIBRARY_PATH=$D/lib/host-math/lib:$D/lib/rocm_sysdeps/lib \
  C_INCLUDE_PATH=$D/lib/rocm_sysdeps/include CPLUS_INCLUDE_PATH=$D/lib/rocm_sysdeps/include \
  CMAKE_PREFIX_PATH=$D/lib/cmake PYTORCH_ROCM_ARCH=gfx950 GPU_ARCHS=gfx950 \
  PATH=$D/bin:$V/bin:/usr/local/bin:/usr/bin:/bin \
  PYTHONPATH=/home/jbowley/repos/gfx950-gluon-tutorials/experiments/hip_version_repro/shim:/home/jbowley/repos/triton_gfx950-tutorial-v2.2/python:/home/jbowley/repos/aiter \
  AITER_JIT_DIR=$X/jit/nightly TRITON_CACHE_DIR=$X/cache/nightly/triton \
  "$@"
