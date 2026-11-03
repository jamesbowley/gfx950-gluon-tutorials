#!/bin/bash
# Compile and time the reduced-grid two-tile points (pilot, then matrix) on HIP 3
# (rocm-smi GPU 1), under the GPU guard. Reruns resume: done points are skipped.
#   bash experiments/streamk_2t_grid/run.sh [pilot|matrix|all]
set -uo pipefail
WHAT=${1:-all}
HERE=$(dirname "$(readlink -f "$0")")
source "$HERE/../../kernels/gemm/intra_wave/a16w16/v31_streamk_aiter/env.sh"
cd "$REPO"
python "$HERE/points.py"
SWEEP=experiments/streamk_2t_grid/sweep_2tg.py
GUARD="python experiments/streamk_v31_rule/gpu_guard.py --smi-index 1 --log $HERE/results/run.guard --"
for name in pilot matrix; do
    [ "$WHAT" = all ] || [ "$WHAT" = "$name" ] || continue
    echo "=== $name: warm $(date)"
    python $SWEEP warm --points "@$HERE/results/$name.txt" --jobs 64 || exit 1
    echo "=== $name: timing $(date)"
    $GUARD python $SWEEP timing --points "@$HERE/results/$name.txt" \
        --out "$HERE/results/$name.csv" || exit 1
done
echo "=== done $(date)"
