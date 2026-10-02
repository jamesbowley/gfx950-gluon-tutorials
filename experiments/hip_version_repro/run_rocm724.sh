#!/bin/bash
# torch 2.10.0+rocm7.2.4.lw.git3d3aa833 on ROCm 7.2.4 (debs unpacked under /opt/rocm-7.2.4),
# with the llirSched Triton and AITER. Clean environment: the container exports ROCm 7.14
# paths from /opt/venv. AMD_COMGR_CACHE=0: with comgr's cache on, the HIP runtime's blit
# kernels are compiled without the device libraries and every GPU call crashes. GPU 0 only.
#   run_rocm724.sh python script.py ...
E=/home/jbowley/rocm724_env
R=/opt/rocm-7.2.4
X=/home/jbowley/repos/gfx950-gluon-tutorials/experiments/hip_version_repro
exec env -i HOME="$HOME" USER="${USER:-root}" TERM="${TERM:-xterm}" \
  HIP_VISIBLE_DEVICES=0 HVR_ENV=rocm724 TMPDIR=/home/jbowley/hvr_tmp/rocm724 AMD_COMGR_CACHE=0 \
  ROCM_PATH=$R ROCM_HOME=$R HIP_PATH=$R LD_LIBRARY_PATH=$R/lib \
  PYTORCH_ROCM_ARCH=gfx950 GPU_ARCHS=gfx950 \
  PATH=$E/venv/bin:$R/bin:$R/llvm/bin:/usr/local/bin:/usr/bin:/bin \
  PYTHONPATH=/home/jbowley/repos/gfx950-gluon-tutorials/experiments/hip_version_repro/shim:/home/jbowley/repos/triton_gfx950-tutorial-v2.2/python:/home/jbowley/repos/aiter \
  AITER_JIT_DIR=$X/jit/rocm724 TRITON_CACHE_DIR=$X/cache/rocm724/triton \
  "$@"
