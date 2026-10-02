#!/bin/bash
# The issue's four tutorial shapes: 3 interleaved rounds x 4 shapes x 4 variants, one process
# per measurement, through run_<env>.sh (GPU 0 only).
#   tutorial.sh <nightly|rocm724>
set -u
ENV=$1
X=/home/jbowley/repos/gfx950-gluon-tutorials/experiments/hip_version_repro
RUN="$X/run_$ENV.sh"
RAW=$X/logs/tutorial_${ENV}_raw.log
OUT=$X/logs/tutorial_${ENV}_results.log
cd /home/jbowley/repos/aiter
BENCH=op_tests/op_benchmarks/triton/bench_gemm_a16w16.py
SHAPES=("4096 4096 8192" "8192 8192 8192" "4096 8192 4096" "4096 106496 16384")
: > $RAW; : > $OUT
$RUN python $X/envinfo.py 2>&1 | tail -1 | tee -a $OUT
last_value() { awk 'NF {v=$NF} END {print v}'; }
for round in 1 2 3; do
  for s in "${SHAPES[@]}"; do
    v=$($RUN python $BENCH --shape $s --metric throughput --backend triton 2>&1 | tee -a $RAW | last_value)
    echo "RESULT triton $s $v round=$round" | tee -a $OUT
    v=$($RUN env AITER_LLIR_SCHED=0 python $BENCH --shape $s --metric throughput --backend gluon --kernel-type compute_bound 2>&1 | tee -a $RAW | last_value)
    echo "RESULT gluon_noplugin $s $v round=$round" | tee -a $OUT
    v=$($RUN python $BENCH --shape $s --metric throughput --backend gluon --kernel-type compute_bound 2>&1 | tee -a $RAW | last_value)
    echo "RESULT gluon_llirsched $s $v round=$round" | tee -a $OUT
    $RUN python $X/hipblaslt_tut.py $s 2>&1 | tee -a $RAW | grep RESULT | sed "s/\$/ round=$round/" | tee -a $OUT
  done
done
echo DONE | tee -a $OUT
