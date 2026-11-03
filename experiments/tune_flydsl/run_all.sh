#!/bin/bash
# Every shape in untuned.csv with a non-empty FlyDSL catalog, one tuner process per
# shape (one process for several shapes runs out of GPU memory: its spawned groups
# leave memory behind, and the 1153x7168x35840 group found 0 bytes free after the
# 640x7168x35840 group). Per-shape outputs go to runs/; a shape whose runs/*_tuned.csv
# exists is skipped, so rerunning resumes. Stops on a foreign GPU process (exit 4).
#   tmux new-session -d -s tune_flydsl "bash experiments/tune_flydsl/run_all.sh; exec bash"
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
cd "$HERE"
mkdir -p runs
log() { echo "$(date '+%F %T') $*" | tee -a tune.log; }
mapfile -t SHAPES < <(python3 -c "
import csv
n = {(r['M'], r['N'], r['K']): int(r['candidates']) for r in csv.DictReader(open('catalog_sizes.csv'))}
for r in csv.DictReader(open('untuned.csv')):
    if n[(r['M'], r['N'], r['K'])]:
        print(r['M'], r['N'], r['K'], n[(r['M'], r['N'], r['K'])])
")
log "start: ${#SHAPES[@]} shapes with FlyDSL candidates"
i=0
for s in "${SHAPES[@]}"; do
    read -r M N K C <<< "$s"
    i=$((i + 1))
    P=runs/${M}x${N}x${K}
    if [ -s "${P}_tuned.csv" ] && [ "$(wc -l < "${P}_tuned.csv")" -ge 2 ]; then
        continue
    fi
    head -1 untuned.csv > "$P.csv"
    echo "$M,$N,$K,False,torch.bfloat16,torch.bfloat16,False,False" >> "$P.csv"
    rm -f "$P.log" "${P}_tuned.csv" "${P}_profile.csv"
    bash run_tuner.sh "$P.csv" "$P" > /dev/null 2>&1
    rc=$?
    log "[$i/${#SHAPES[@]}] ${M}x${N}x${K} ($C candidates): exit $rc in $(( $(cat "$P.end") - $(cat "$P.start") )) s"
    if [ $rc -eq 4 ]; then
        log "foreign process on the GPU; stopping (rerun to resume)"
        rm -f "${P}_tuned.csv"
        exit 4
    fi
done
log "done"
