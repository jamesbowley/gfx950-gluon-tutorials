#!/bin/bash
# The two-stage session on physical GPU 2 (HIP 2), unattended and resumable:
#   1. split_sweep.py run      per-tile split tables (stages A, B1, B2, N), unless already done
#   2. split_report.py         results/split/tables.json, tables.md
#   3. split_extra.py run      audit and spot checks (AUD, GEO, ORI, WAV); extra_report.md
#   4. tilesel.py run          tile selection on the tables' picks; tilesel_report.md
# WAIT_PID: wait for an already running step 1 (that process) instead of starting it.
# Logs: tmp/split_*.log. Every step skips what its results already hold.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TRITON_CACHE_DIR=$HERE/tmp/triton_split
unset SPLIT_OUT
[ "$HIP_VISIBLE_DEVICES" = 2 ] || { echo "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES, expected 2"; exit 1; }
if [ -n "$WAIT_PID" ]; then
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    grep -q "all stages done" "$HERE/tmp/split_run.log" || { echo "split_sweep.py did not finish"; exit 1; }
else
    python v27_streamk_tile/split_sweep.py run --jobs 32 >> "$HERE/tmp/split_run.log" 2>&1
fi
python v27_streamk_tile/split_report.py > "$HERE/tmp/split_report.log" 2>&1
python v27_streamk_tile/split_extra.py run --jobs 32 > "$HERE/tmp/split_extra.log" 2>&1
python v27_streamk_tile/split_extra.py report >> "$HERE/tmp/split_extra.log" 2>&1
python v27_streamk_tile/tilesel.py run --jobs 32 > "$HERE/tmp/split_tilesel.log" 2>&1
python v27_streamk_tile/tilesel.py report >> "$HERE/tmp/split_tilesel.log" 2>&1
echo "session done: $(date)"
