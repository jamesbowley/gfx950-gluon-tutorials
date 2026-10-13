#!/bin/bash
# Build hbl_bench once per hipBLASLt version, each with its own hipcc, headers and libhipblaslt (rpath).
set -e
I=/home/jbowley/repos/gfx950-gluon-tutorials/experiments/hip_version_repro/hipblaslt_issue
cd $I
R=/opt/rocm-7.2.4
$I/run_rocm724_gpu3.sh hipcc -O2 -std=c++17 --offload-arch=gfx950 hbl_bench.cpp -I$R/include \
  -L$R/lib -lhipblaslt -Wl,-rpath,$R/lib -o hbl_bench_1.2.2
for pair in \
            "nightly:/home/jbowley/venvs/rocm-nightly-20260929/lib/python3.12/site-packages:1.5.0"; do
  IFS=: read env S ver <<< "$pair"
  $I/run_${env}_gpu3.sh hipcc -O2 -std=c++17 --offload-arch=gfx950 hbl_bench.cpp -I$S/_rocm_sdk_devel/include \
    -L$S/_rocm_sdk_libraries/lib -L$S/_rocm_sdk_core/lib -lhipblaslt \
    -Wl,-rpath,$S/_rocm_sdk_libraries/lib -Wl,-rpath,$S/_rocm_sdk_core/lib -o hbl_bench_$ver
done
ls -la hbl_bench_*
# TheRock tarballs: ROCm 10.0.0 (hipBLASLt 1.4.1) and the 10.2.0a20260929 nightly (hipBLASLt 1.5.0, same build as the venv).
for pair in "rocm1000:10.0.0:10.0.0" "therock_nightly:10.2.0a20260929:1.5.0t"; do
  IFS=: read env ver tag <<< "$pair"
  P=/home/jbowley/therock/$ver
  $I/run_${env}_gpu3.sh hipcc -O2 -std=c++17 --offload-arch=gfx950 hbl_bench.cpp -I$P/include \
    -L$P/lib -lhipblaslt -Wl,-rpath,$P/lib -o hbl_bench_$tag
done
ls -la hbl_bench_*
