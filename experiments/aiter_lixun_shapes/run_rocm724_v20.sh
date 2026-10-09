#!/bin/bash
# hip_version_repro/run_rocm724.sh for the v20 comparison: torch 2.10.0+rocm7.2.4.lw.git3d3aa833
# (hipBLASLt 1.2.2) on ROCm 7.2.4 in a clean environment, one GPU: HIP device $V20CMP_GPU, default 0
# (PCI bus 117, as on Sep 29); run_v20_vs_aiter.py checks the device's PCI bus (GPUS there).
# AMD_COMGR_CACHE=0: with comgr's cache on, the HIP runtime's blit kernels are compiled without
# the device libraries and every GPU call crashes. No PYTHONPATH here: run_v20_vs_aiter.py gives
# each of its two workers its own (AITER worktree + AITER's llirSched, or the tutorial stack).
#   run_rocm724_v20.sh python run_v20_vs_aiter.py sweep ...
E=/home/jbowley/rocm724_env
R=/opt/rocm-7.2.4
exec env -i HOME="$HOME" USER="${USER:-root}" TERM="${TERM:-xterm}" \
  HIP_VISIBLE_DEVICES=${V20CMP_GPU:-0} V20CMP_ENV=rocm724 TMPDIR=/home/jbowley/hvr_tmp/rocm724 AMD_COMGR_CACHE=0 \
  ROCM_PATH=$R ROCM_HOME=$R HIP_PATH=$R LD_LIBRARY_PATH=$R/lib \
  PYTORCH_ROCM_ARCH=gfx950 GPU_ARCHS=gfx950 \
  PATH=$E/venv/bin:$R/bin:$R/llvm/bin:/usr/local/bin:/usr/bin:/bin \
  "$@"
