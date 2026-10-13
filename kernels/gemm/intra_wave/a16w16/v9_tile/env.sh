# Environment for the v9_tile runs: Triton gfx950-tutorial-v2.2 + deepbind shim, the shipping
# llir+amdgcnas stack, GPU 2 unless $GPU says otherwise.
REPO=/home/jbowley/repos/gfx950-gluon-tutorials
export PYTHONPATH=$REPO/scripts/triton_deepbind_shim:/home/jbowley/repos/triton_gfx950-tutorial-v2.2/python
export LLVM_PASS_PLUGIN_PATH=$REPO/plugins/llir_scheduler/libLlirSched.so
export TRITON_AMDGCNAS_PLUGIN=1
export HIP_VISIBLE_DEVICES=${GPU:-2}
cd $REPO/kernels/gemm/intra_wave/a16w16
