# Environment for the v31 runs: v29's stack (the installed Triton with the MFMA scheduler, no plugins,
# the deepbind shim) plus the AITER checkout on PYTHONPATH (the kernel imports AITER's device_info and
# graph_alloc). HIP 3 unless $GPU says otherwise: HIP 3 is rocm-smi GPU 1 (PCI 0000:15:00.0).
REPO=/home/jbowley/repos/gfx950-gluon-tutorials
export PYTHONPATH=$REPO/scripts/triton_deepbind_shim:/home/jbowley/repos/aiter
unset LLVM_PASS_PLUGIN_PATH
unset TRITON_AMDGCNAS_PLUGIN
export HIP_VISIBLE_DEVICES=${GPU:-3}
cd $REPO/kernels/gemm/intra_wave/a16w16
