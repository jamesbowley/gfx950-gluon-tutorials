#!/bin/bash
# After run_all.sh logs "done": re-time the FlyDSL points (each shape's top 5 from AITER's
# profile, plus the large-GEMM config on the 20 empty-catalog shapes) in the
# tune_aiter_shapes harness on rocm-smi GPU 1, then regenerate the report.
#   tmux new-session -d -s retime_flydsl "bash experiments/tune_flydsl/run_retime.sh; exec bash"
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
until grep -q " done$" "$HERE/tune.log" 2>/dev/null; do
    if grep -q "stopping (rerun to resume)" "$HERE/tune.log" 2>/dev/null; then
        echo "tuner stopped early; not re-timing"; exit 4
    fi
    sleep 60
done
source "$HERE/../../kernels/gemm/intra_wave/a16w16/v31_streamk_aiter/env.sh"
cd "$REPO"
export FLYDSL_RUNTIME_CACHE_DIR=$HERE/cache
T=experiments/tune_aiter_shapes
python experiments/streamk_v31_rule/gpu_guard.py --smi-index 1 --log $T/results/retime_flydsl.log.guard -- \
    python $T/tune.py timing 2>&1 | tee -a $T/results/retime_flydsl.log
python $T/tune.py report 2>&1 | tee -a $T/results/retime_flydsl.log
echo "RETIME DONE" | tee -a $T/results/retime_flydsl.log
