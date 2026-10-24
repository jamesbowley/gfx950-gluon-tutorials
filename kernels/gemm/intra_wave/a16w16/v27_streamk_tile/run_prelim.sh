#!/bin/bash
# bash v27_streamk_tile/run_prelim.sh: the preliminary sweep (WORKLOG §11) on GPUS (default 0-7), or another
# sweep.py preset as PRESET.
# First every v27 build is compiled into Triton's cache without launching (sweep.py warm, JOBS
# processes spread over the GPUs), then one timing shard per GPU writes results/<preset>_<i>.csv (resumable: a
# re-run skips the points already there), and the shards are merged into results/<preset>.csv. Logs go to
# tmp/<preset>_*.log.
source "$(dirname "$0")/env.sh"
PRESET=${PRESET:-PRELIM}
TAG=$(echo "$PRESET" | tr '[:upper:]' '[:lower:]')
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
IFS=, read -ra G <<< "$GPUS"
N=${#G[@]}
R=v27_streamk_tile/results
T=v27_streamk_tile/tmp
mkdir -p $T
python v27_streamk_tile/sweep.py warm --points $PRESET --jobs ${JOBS:-48} --gpus "$GPUS" > $T/${TAG}_warm.log 2>&1
tail -1 $T/${TAG}_warm.log
for i in "${!G[@]}"; do
    HIP_VISIBLE_DEVICES=${G[$i]} python v27_streamk_tile/sweep.py --points $PRESET --shard $i/$N \
        --out $R/${TAG}_$i.csv > $T/${TAG}_$i.log 2>&1 &
done
wait
head -1 $R/${TAG}_0.csv > $R/$TAG.csv
for i in "${!G[@]}"; do
    tail -n +2 $R/${TAG}_$i.csv >> $R/$TAG.csv
done
echo "$(($(wc -l < $R/$TAG.csv) - 1)) rows in $R/$TAG.csv"
