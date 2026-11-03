#!/bin/bash
# The whole FlyDSL run, meant for a detached tmux session (survives closing the editor):
#   tmux new-session -d -s flydsl "bash experiments/tune_flydsl/run_pipeline.sh; exec bash"
# 1. inputs from the current AITER's FlyDSL policy (make_inputs.py), unless present;
# 2. a one-shape check (2561x6288x7168) that the tuner times FlyDSL (no 0 us) and that
#    the kernel names parse in tune_aiter_shapes/tune.py;
# 3. every shape (run_all.sh, resumable); 4. re-time in our harness and report (run_retime.sh).
# Progress: pipeline.log here, tune.log (per shape), ../tune_aiter_shapes/results/retime_flydsl.log.
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
cd "$HERE"
log() { echo "$(date '+%F %T') $*" | tee -a pipeline.log; }
source "$HERE/../../kernels/gemm/intra_wave/a16w16/v31_streamk_aiter/env.sh"
cd "$HERE"
log "AITER $(git -C /home/jbowley/repos/aiter log -1 --format='%h %cd') flydsl $(pip show flydsl 2>/dev/null | awk '/^Version/{print $2}')"

if [ ! -s untuned.csv ] || [ ! -s catalog_sizes.csv ]; then
    log "generating inputs"
    python make_inputs.py 2>&1 | grep -v "^\[aiter\]" | tee -a pipeline.log
fi

if [ ! -s check_tuned.csv ]; then
    log "one-shape check"
    head -1 untuned.csv > check.csv
    grep "^2561,6288,7168," untuned.csv >> check.csv
    bash run_tuner.sh check.csv check > /dev/null 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then log "check: tuner exit $rc (see check.log); stopping"; exit $rc; fi
    python3 - <<'EOF' 2>&1 | tee -a pipeline.log
import csv, sys
sys.path.insert(0, "../tune_aiter_shapes")
rows = list(csv.DictReader(open("check_profile.csv")))
us = [float(r["us"]) for r in rows]
bad = sum(u <= 0 or u == float("inf") for u in us)
print(f"check: {len(rows)} candidates, {bad} with no time, fastest {min(us):.1f} us")
import tune
unparsed = [r["kernelName"] for r in rows if not tune.FLY_NAME.fullmatch(r["kernelName"])]
print(f"check: {len(unparsed)} kernel names tune.py cannot parse" + (f", e.g. {unparsed[0]}" if unparsed else ""))
sys.exit(1 if bad == len(rows) or unparsed else 0)
EOF
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        rm -f check_tuned.csv
        log "check failed; stopping"
        exit 1
    fi
fi

log "tuning every shape"
bash run_all.sh
rc=$?
if [ $rc -ne 0 ]; then log "run_all.sh exit $rc; stopping (rerun this script to resume)"; exit $rc; fi
log "re-timing in our harness"
bash run_retime.sh
log "pipeline finished"
