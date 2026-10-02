#!/bin/bash
# Heuristic top-1 on all 630 shapes, each hipBLASLt version in turn, GPU 3 only.
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
  IFS=: read ver env <<< "$v"
  wait_idle
  echo "start $ver $(date -u +%T)"
  ./run_${env}_gpu3.sh ./hbl_bench_$ver shapes_all.txt heuristic > heuristic_$ver.csv 2> heuristic_$ver.err
  echo "done $ver $(date -u +%T) rows=$(grep -c -v '^#' heuristic_$ver.csv)"
done
echo STAGEA_DONE
