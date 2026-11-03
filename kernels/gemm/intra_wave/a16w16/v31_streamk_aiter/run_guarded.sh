#!/bin/bash
# Run a command under the GPU guard on rocm-smi GPU 1 (HIP 3), logging to LOG; paths
# are relative to the caller's directory.
#   bash v31_streamk_aiter/run_guarded.sh LOG CMD ...
set -uo pipefail
HERE_CWD=$PWD
source "$(dirname "$(readlink -f "$0")")/env.sh"
cd "$HERE_CWD"
LOG=$1; shift
python $REPO/experiments/streamk_v31_rule/gpu_guard.py --smi-index 1 --log "$LOG.guard" -- "$@" 2>&1 | tee -a "$LOG"
exit ${PIPESTATUS[0]}
