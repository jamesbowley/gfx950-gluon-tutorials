# Environment for the v27 runs (and v26's re-baseline): the gfx950-tutorial-v3.0 stack (the installed
# Triton, a build that contains 3b0c7f081: the MFMA scheduler, enabled by matmul()'s schedule_hint),
# no plugins, with the deepbind shim (pip-installed ROCm), GPU 2 unless $GPU says otherwise. This is
# v9_tile/env.sh without amdgcnas: v9_tile WORKLOG §16's stack. Without amdgcnas LLVM's hazard s_nops
# stay in, so v9_tile_small needs no plugin skip.
REPO=/home/jbowley/repos/gfx950-gluon-tutorials
export PYTHONPATH=$REPO/scripts/triton_deepbind_shim
unset LLVM_PASS_PLUGIN_PATH
unset TRITON_AMDGCNAS_PLUGIN
export HIP_VISIBLE_DEVICES=${GPU:-2}
cd $REPO/kernels/gemm/intra_wave/a16w16
