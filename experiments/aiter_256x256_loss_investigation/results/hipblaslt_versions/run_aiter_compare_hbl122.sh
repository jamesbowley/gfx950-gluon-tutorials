#!/bin/bash
# run_aiter_compare.py with hipBLASLt 1.2.2 + rocBLAS/hipBLAS from ROCm 7.2.4 swapped into the
# TheRock 7.14 torch in /opt/venv (as run_hbl122.sh), plus the Triton/AITER path the Gluon
# kernels need. Writes results/aiter_live_compare_hbl122.{csv,md}.
#   HIP_VISIBLE_DEVICES=0 run_aiter_compare_hbl122.sh [--shapes N] [--rounds 3]
HERE=$(cd "$(dirname "$0")" && pwd)
H=/home/jbowley/rocm724_env/hbl122/lib_min
SP=/opt/venv/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$H:$SP/_rocm_sdk_core/lib:$SP/_rocm_sdk_core/lib/host-math/lib:$SP/_rocm_sdk_libraries/lib:$LD_LIBRARY_PATH
export LD_PRELOAD="$H/libhipblaslt.so.1 $H/librocblas.so.5 $H/libhipblas.so.3"
export PYTHONPATH=$HERE/hbl122_shim:/home/jbowley/repos/triton_gfx950-tutorial-v2.2/python:/home/jbowley/repos/aiter
exec /opt/venv/bin/python "$HERE/../../run_aiter_compare.py" --tag hbl122 "$@"
