#!/bin/bash
# ROCm 10.0.0 (hipBLASLt 1.4.1, therock-10.0), GPU 3 only: 630 shapes heuristic at 128 MiB and 32 MiB,
# then the recomputed regression set: best256 in 10.0.0 and 1.5.0, ANALYTICAL_GEMM_PICK in 1.5.0.
I=/home/jbowley/repos/gfx950-gluon-tutorials/experiments/hip_version_repro/hipblaslt_issue
cd $I
wait_idle() {
  local n=0
  while [ $n -lt 5 ]; do
    u=$(rocm-smi -d 3 --showuse 2>/dev/null | awk -F': ' '/GPU use/{print $NF}' | tr -d ' %')
    if [ "$u" = "0" ]; then n=$((n+1)); else n=0; echo "GPU 3 busy ($u%), waiting"; fi
    sleep 2
  done
}
wait_idle; echo "heuristic 10.0.0 $(date -u +%T)"
./run_rocm1000_gpu3.sh ./hbl_bench_10.0.0 shapes_all.txt heuristic > heuristic_10.0.0.csv
wait_idle; echo "ws32 10.0.0 $(date -u +%T)"
./run_rocm1000_gpu3.sh env HBL_WORKSPACE_KB=32768 ./hbl_bench_10.0.0 shapes_all.txt heuristic > heuristic_ws32_10.0.0.csv
python3 analyze.py stageA
wait_idle; echo "best256 10.0.0 $(date -u +%T)"
./run_rocm1000_gpu3.sh ./hbl_bench_10.0.0 regress_shapes.txt best256 > best256_10.0.0.csv
wait_idle; echo "best256 1.5.0 $(date -u +%T)"
./run_nightly_gpu3.sh ./hbl_bench_1.5.0 regress_shapes.txt best256 > best256_1.5.0.csv
wait_idle; echo "pick256 1.5.0 $(date -u +%T)"
./run_nightly_gpu3.sh env ANALYTICAL_GEMM_PICK=256x256x64 ./hbl_bench_1.5.0 regress_shapes.txt heuristic > pick256_1.5.0.csv
echo "STAGE1000_DONE $(date -u +%T)"
