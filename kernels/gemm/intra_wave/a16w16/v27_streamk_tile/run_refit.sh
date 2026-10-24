#!/bin/bash
# bash v27_streamk_tile/run_refit.sh: a refit.py study (STUDY, default refit; PREFIX, default the study) on GPUS
# (default 0-3), resumable step by step:
# 1. refit.py init (the run manifest), unless results/<prefix>_manifest.json exists;
# 2. sweep.py warm over the manifest's points (JOBS processes, its own Triton cache tmp/triton_<prefix>), with the
#    failures in results/<prefix>_warm_fail.json, unless that exists;
# 3. rounds 1 and 2, one shard per GPU (refit.py shard; a re-run skips the points already timed);
# 4. refit.py close (round 3's points), unless results/<prefix>_round3.json exists; round 3;
# 5. refit.py merge: results/<prefix>.csv and results/<prefix>_report.md.
# SINGLE_GPU=g runs each round's slots one after another on GPU g (the manifest's gpus all g); FROM_ROUND=r skips the
# rounds before r (whose files are named after the GPUs they ran on).
# A shard that stops (a frozen file changed: exit 3; foreign processes stayed on its GPU FOREIGN_WAIT seconds, default
# 1800: exit 4) stops the run. Logs go to tmp/<prefix>_*.log.
source "$(dirname "$0")/env.sh"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
# A GPU memory fault writes a 6-58 GB core dump (gpucore.<pid>.gpu) per faulting worker; 16 filled the disk once.
ulimit -c 0
STUDY=${STUDY:-refit}
P=${PREFIX:-$STUDY}
GPUS=${GPUS:-0,1,2,3}
IFS=, read -ra G <<< "$GPUS"
R=v27_streamk_tile/results
T=v27_streamk_tile/tmp
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$PWD/$T/triton_$P}
mkdir -p $T "$TRITON_CACHE_DIR"
set -o pipefail

if [ ! -f $R/${P}_manifest.json ]; then
    python v27_streamk_tile/refit.py init --study $STUDY --prefix $P --gpus "$GPUS" ${INIT_ARGS} \
        | tee $T/${P}_init.log || exit 1
fi
if [ ! -f $R/${P}_warm_fail.json ]; then
    python v27_streamk_tile/sweep.py warm --points @$R/${P}_manifest.json --frozen $R/${P}_manifest.json \
        --jobs ${JOBS:-48} --gpus "$GPUS" --fail-out $R/${P}_warm_fail.json > $T/${P}_warm.log 2>&1 || exit 1
    tail -1 $T/${P}_warm.log
fi

run_round() {
    local pids=() rc=0
    if [ -n "$SINGLE_GPU" ]; then
        # One GPU: the manifest's slots one after another (its gpus all SINGLE_GPU, GPUS as many entries).
        for i in "${!G[@]}"; do
            HIP_VISIBLE_DEVICES=$SINGLE_GPU python v27_streamk_tile/refit.py shard --prefix $P --round $1 --slot $i \
                --foreign-wait ${FOREIGN_WAIT:-1800} > $T/${P}_$1_slot$i.log 2>&1 || { rc=$?; break; }
        done
        echo "round $1 done (exit $rc): $(date)"
        return $rc
    fi
    for i in "${!G[@]}"; do
        HIP_VISIBLE_DEVICES=${G[$i]} python v27_streamk_tile/refit.py shard --prefix $P --round $1 --slot $i \
            --foreign-wait ${FOREIGN_WAIT:-1800} > $T/${P}_$1_${G[$i]}.log 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do
        wait $p || rc=$?
    done
    echo "round $1 done (exit $rc): $(date)"
    return $rc
}

for r in 1 2; do
    [ $r -lt ${FROM_ROUND:-1} ] && continue
    run_round $r || exit 1
done
if [ ! -f $R/${P}_round3.json ]; then
    python v27_streamk_tile/refit.py close --prefix $P | tee $T/${P}_close.log || exit 1
fi
run_round 3 || exit 1
python v27_streamk_tile/refit.py merge --prefix $P | tee $T/${P}_merge.log
