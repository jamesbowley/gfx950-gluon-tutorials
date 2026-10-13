#!/bin/bash
# Everything on GPU 0, one environment at a time: tutorial shapes, then the 630-shape sweep.
X=/home/jbowley/repos/gfx950-gluon-tutorials/experiments/hip_version_repro
cd $X
wait_idle() {
  # GPU 0 must read 0% busy on 5 consecutive 2 s samples; wait otherwise, never switch GPU.
  local n=0
  while [ $n -lt 5 ]; do
    u=$(rocm-smi -d 0 --showuse 2>/dev/null | awk -F': ' '/GPU use/{print $NF}' | tr -d ' %')
    if [ "$u" = "0" ]; then n=$((n+1)); else n=0; echo "GPU 0 busy ($u%), waiting"; fi
    sleep 2
  done
  echo "GPU 0 idle at $(date -u +%FT%TZ)"
}
for e in nightly rocm724; do wait_idle; ./tutorial.sh $e > logs/tutorial_$e.log 2>&1; done
for e in nightly rocm724; do wait_idle; ./run_$e.sh python sweep.py $e > logs/sweep_$e.log 2>&1; done
echo ALLDONE
