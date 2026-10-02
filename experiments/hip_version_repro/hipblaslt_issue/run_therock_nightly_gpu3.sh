#!/bin/bash
# TheRock gfx950 tarball ROCm 10.2.0a20260929 (runtime + tests: hipblaslt-bench), clean environment, GPU 3 only.
#   run_therock_nightly_gpu3.sh <command> ...
P=/home/jbowley/therock/10.2.0a20260929
exec env -i HOME="$HOME" USER="${USER:-root}" TERM="${TERM:-xterm}" \
  HIP_VISIBLE_DEVICES=3 HVR_ENV=therock_nightly TMPDIR=/home/jbowley/hvr_tmp/therock_nightly \
  ROCM_PATH=$P ROCM_HOME=$P HIP_PATH=$P LD_LIBRARY_PATH=$P/lib \
  HIP_DEVICE_LIB_PATH=$P/lib/llvm/amdgcn/bitcode \
  PATH=$P/bin:$P/lib/llvm/bin:/usr/local/bin:/usr/bin:/bin \
  "$@"
