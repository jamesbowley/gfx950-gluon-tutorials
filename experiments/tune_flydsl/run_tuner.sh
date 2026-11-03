#!/bin/bash
# AITER's a16w16 tuner, FlyDSL only, on rocm-smi GPU 1 (HIP 3) under the GPU guard:
#   bash run_tuner.sh INPUT.csv PREFIX     -> PREFIX_tuned.csv, PREFIX_profile.csv, PREFIX.log
# env.sh's PYTHONPATH plus pyhook/ (see pyhook/sitecustomize.py); the compile cache is cache/.
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
source "$HERE/../../kernels/gemm/intra_wave/a16w16/v31_streamk_aiter/env.sh"
cd "$HERE"
export PYTHONPATH=$HERE/pyhook:$PYTHONPATH FLYDSL_RUNTIME_CACHE_DIR=$HERE/cache
IN=$1; P=$2
date +%s > "$P.start"
python "$REPO/experiments/streamk_v31_rule/gpu_guard.py" --smi-index 1 --log "$P.log.guard" -- \
  python3 /home/jbowley/repos/aiter/csrc/gemm_a16w16/gemm_a16w16_tune.py -i "$IN" -o "${P}_tuned.csv" \
  -o2 "${P}_profile.csv" --libtype flydsl --mp 1 --timeout 14400 >> "$P.log" 2>&1
rc=$?
date +%s > "$P.end"
echo "EXIT $rc" | tee -a "$P.log"
exit $rc
