# Environment for the write-through sync before/after runs: GPU 0, Triton v2.2 + deepbind shim,
# llirSched plugin, no amdgcnas hook (bench.py installs it for any non-empty value).
REPO=/home/jbowley/repos/gfx950-gluon-tutorials
export PYTHONPATH=$REPO/scripts/triton_deepbind_shim:/home/jbowley/repos/triton_gfx950-tutorial-v2.2/python
export LLVM_PASS_PLUGIN_PATH=$REPO/plugins/llir_scheduler/libLlirSched.so
unset TRITON_AMDGCNAS_PLUGIN
export HIP_VISIBLE_DEVICES=0
cd $REPO/kernels/gemm/intra_wave/a16w16
echo "=== env $(date -u +%FT%TZ)"
env | grep -E 'TRITON_AMDGCNAS|LLVM_PASS|HIP_VISIBLE|PYTHONPATH' | sort
rocm-smi -d 0 --showuse --showpids 2>/dev/null | grep -E 'GPU use|PID|^[0-9]' | head -5
