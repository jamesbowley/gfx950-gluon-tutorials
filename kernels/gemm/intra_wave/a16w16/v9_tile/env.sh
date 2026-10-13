# Environment for the v9_tile runs: the gfx950-tutorial-v3.0 stack (the installed Triton, a build
# that contains 3b0c7f081: the MFMA scheduler, enabled by matmul()'s schedule_hint) + amdgcnas,
# with the deepbind shim (pip-installed ROCm), GPU 2 unless $GPU says otherwise. Sections 1-14 of
# WORKLOG.md were measured on the earlier stack (Triton gfx950-tutorial-v2.2 + the llirSched plugin
# + amdgcnas); since amdgcnas moved with the v3.0 re-pin, that stack gives wrong results and is no
# longer set up here. plugins/amdgcnas is main's, without the fixes of WORKLOG §2 and §14.3: it
# miscompiled v9_tile's former hand-built layouts (results/check_tiles_v30.md); the derived layouts
# pass under it (WORKLOG §15.5). `unset TRITON_AMDGCNAS_PLUGIN` after sourcing for the scheduler alone.
REPO=/home/jbowley/repos/gfx950-gluon-tutorials
export PYTHONPATH=$REPO/scripts/triton_deepbind_shim
unset LLVM_PASS_PLUGIN_PATH
export TRITON_AMDGCNAS_PLUGIN=1
export HIP_VISIBLE_DEVICES=${GPU:-2}
cd $REPO/kernels/gemm/intra_wave/a16w16
