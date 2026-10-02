#!/bin/bash
# hipBLASLt 1.2.2 + rocBLAS/hipBLAS from ROCm 7.2.4, on the TheRock 7.14 torch/HIP runtime in /opt/venv.
# The ROCm 7.2.4 HIP runtime itself cannot build its blit kernels in this container, so only the
# BLAS libraries (and their own kernel files) are swapped in.
#   run_hbl122.sh script.py args...
H=/home/jbowley/rocm724_env/hbl122/lib_min
SP=/opt/venv/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$H:$SP/_rocm_sdk_core/lib:$SP/_rocm_sdk_libraries/lib:$LD_LIBRARY_PATH
export LD_PRELOAD="$H/libhipblaslt.so.1 $H/librocblas.so.5 $H/libhipblas.so.3"
unset PYTHONPATH
exec /opt/venv/bin/python -c "import rocm_sdk, runpy, sys; rocm_sdk.preload_libraries = lambda *a, **k: None; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name=\"__main__\")" "$@"
