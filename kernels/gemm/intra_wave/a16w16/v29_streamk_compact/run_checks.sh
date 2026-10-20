#!/bin/bash
# v29 stress checks on HIP 1 (physical GPU 0): every ending, exchange and policy, with and without bias, on the
# quadrant and small tiles, the no-full-tile (compact launch) sets, and the stale-count negative control.
#   bash v29_streamk_compact/run_checks.sh
# Results: v29_streamk_compact/results/checks/<tile>[_nofull].md, a summary line per run in summary.txt.
source "$(dirname "$0")/env.sh"
TILES=${TILES:-"256x256x64 256x128x64 128x256x64 128x128x64 128x128x128 16x256x128 32x128x128 128x32x128 16x16x256"}
NOFULL=${NOFULL:-"256x256x64 128x128x128 16x256x128"}
OUT=${OUT:-v29_streamk_compact/results/checks}
mkdir -p $OUT
log() { echo "$(date +%H:%M:%S) $1: $(tail -1 $2) ($(grep -c '^| [0-9]' $2) configurations)" | tee -a $OUT/summary.txt; }
for t in $TILES; do
  python v29_streamk_compact/check_streamk.py --tile $t --jobs 24 > $OUT/$t.md 2>&1
  log $t $OUT/$t.md
done
for t in $NOFULL; do
  python v29_streamk_compact/check_streamk.py --tile $t --no-full --jobs 24 > $OUT/${t}_nofull.md 2>&1
  log "$t no-full" $OUT/${t}_nofull.md
done
python v29_streamk_compact/check_streamk.py --tile 256x128x64 --stale-count --launches 3 --timeout 120 \
  > $OUT/stale_256x128x64.md 2>&1
log "256x128x64 stale-count (must FAIL)" $OUT/stale_256x128x64.md
