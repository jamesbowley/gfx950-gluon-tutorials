# Environment for the v9_kreverse runs: the installed Triton (gfx950-tutorial-v3.0 stack) + amdgcnas
# (the shipping llir+amdgcnas config), with the deepbind shim, on physical GPU 1 (amd-smi / rocm-smi
# index 1, BDF 0000:15:00.0) unless $SMI_GPU says otherwise; GPU 0 is in use. HIP numbers the
# devices in another order on this machine (HIP 1 is BDF 0000:05:00.0, physical GPU 0), so the
# HIP index is looked up from the BDF instead of reusing the amd-smi index.
REPO=/home/jbowley/repos/gfx950-gluon-tutorials
export PYTHONPATH=$REPO/scripts/triton_deepbind_shim
unset LLVM_PASS_PLUGIN_PATH
export TRITON_AMDGCNAS_PLUGIN=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export SMI_GPU=${SMI_GPU:-1}
export GPU_BDF=$(amd-smi list -g "$SMI_GPU" 2>/dev/null | awk '/BDF/ {print tolower($2)}')
export HIP_VISIBLE_DEVICES=$(python - <<EOF
import torch
bus = int("$GPU_BDF".split(":")[1], 16)
print([i for i in range(torch.cuda.device_count()) if torch.cuda.get_device_properties(i).pci_bus_id == bus][0])
EOF
)
echo "v9_kreverse env: amd-smi GPU $SMI_GPU, BDF $GPU_BDF, HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES" >&2
cd $REPO/kernels/gemm/intra_wave/a16w16
