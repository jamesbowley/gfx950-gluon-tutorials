#!/bin/bash
# GPU 3 only, one run at a time:
#  1. heuristic top-1 on all 630 shapes with a 32 MiB workspace, all three versions;
#  2. on the >=5% regression set, in 1.4.1 and 1.5.0: best MT256x256x64 solution (128 MiB)
#     and heuristic with ANALYTICAL_GEMM_PICK=256x256x64 (128 MiB).
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
for v in 1.2.2:rocm724 1.4.1:rocm714 1.5.0:nightly; do
  IFS=: read ver env <<< "$v"; wait_idle; echo "ws32 $ver $(date -u +%T)"
  ./run_${env}_gpu3.sh env HBL_WORKSPACE_KB=32768 ./hbl_bench_$ver shapes_all.txt heuristic > heuristic_ws32_$ver.csv 2> heuristic_ws32_$ver.err
done
for v in 1.4.1:rocm714 1.5.0:nightly; do
  IFS=: read ver env <<< "$v"; wait_idle; echo "pick256 $ver $(date -u +%T)"
  ./run_${env}_gpu3.sh env ANALYTICAL_GEMM_PICK=256x256x64 ./hbl_bench_$ver regress_shapes.txt heuristic > pick256_$ver.csv 2> pick256_$ver.err
  wait_idle; echo "best256 $ver $(date -u +%T)"
  ./run_${env}_gpu3.sh ./hbl_bench_$ver regress_shapes.txt best256 > best256_$ver.csv 2> best256_$ver.err
done
echo "STAGEB_DONE $(date -u +%T)"
