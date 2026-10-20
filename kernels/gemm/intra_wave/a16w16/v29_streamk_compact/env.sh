# Environment for the v29 runs: v28's stack (the gfx950-tutorial-v3.0 stack, the installed Triton with the
# MFMA scheduler, no plugins, the deepbind shim). HIP 1 unless $GPU says otherwise: HIP 1 is amd-smi GPU 0
# (PCI 0000:05:00.0), the only GPU v29's trials may use.
REPO=/home/jbowley/repos/gfx950-gluon-tutorials
export PYTHONPATH=$REPO/scripts/triton_deepbind_shim
unset LLVM_PASS_PLUGIN_PATH
unset TRITON_AMDGCNAS_PLUGIN
export HIP_VISIBLE_DEVICES=${GPU:-1}
cd $REPO/kernels/gemm/intra_wave/a16w16
