#!/bin/bash
# GPUS=0,1 bash v27_streamk_tile/run_check_grid.sh TILE [TILE ...]: check_grid.py per tile, the tiles dealt to
# the GPUs in turn and run one after the other on each, to results/check_grid_<tile>.md. CHECK=check_streamk
# (and ARGS, e.g. "--no-full") runs check_streamk.py the same way, to results/check_streamk_<tile><SUFFIX>.md.
source "$(dirname "$0")/env.sh"
CHECK=${CHECK:-check_grid}
IFS=, read -ra G <<< "${GPUS:-0,1,2,3,4,5,6,7}"
TILES=("$@")
for i in "${!G[@]}"; do
    (
        for ((t = i; t < ${#TILES[@]}; t += ${#G[@]})); do
            tile=${TILES[$t]}
            HIP_VISIBLE_DEVICES=${G[$i]} python v27_streamk_tile/$CHECK.py --tile "$tile" --jobs ${JOBS:-12} $ARGS \
                > "v27_streamk_tile/results/${CHECK}_$tile${SUFFIX}.md" 2> "v27_streamk_tile/tmp/${CHECK}_$tile${SUFFIX}.err"
        done
    ) &
done
wait
