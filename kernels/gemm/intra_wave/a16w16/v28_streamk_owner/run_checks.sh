#!/bin/bash
# Owner-ending stress checks on HIP 1 (physical GPU 0), of v28 itself ("v28") or of the archived experiment kernel's
# STREAMK_OWNER_OPT variants (experiments/matmul_kernel_variants.py, the code WORKLOG sections 1-9 timed):
#   bash v28_streamk_owner/run_checks.sh v28 full      # every partial type
#   bash v28_streamk_owner/run_checks.sh "b bs" fp32   # archive variants
# Results: v28_streamk_owner/results/checks/<variant>_<tile>.md (and _nofull); a summary line per run in
# results/checks/summary.txt. Do not edit matmul_kernel.py while this runs: every configuration imports it afresh.
source "$(dirname "$0")/env.sh"
VARIANTS=${1:-"v28"}
SCOPE=${2:-full}
TILES=${TILES:-"256x256x64 256x128x64 128x256x64 128x128x64 128x128x128"}
OUT=${OUT:-v28_streamk_owner/results/checks}
mkdir -p $OUT
PARTIALS=""
[ "$SCOPE" = fp32 ] && PARTIALS="--partials fp32"
run_env() {  # VARIANT CMD...: v28 itself, or the archive with STREAMK_OWNER_OPT=VARIANT
  local v=$1; shift
  if [ "$v" = v28 ]; then "$@"; else
    V28_MATMUL_MODULE=v28_streamk_owner.experiments.matmul_kernel_variants STREAMK_OWNER_OPT=$v "$@"; fi
}
for v in $VARIANTS; do
  for t in $TILES; do
    f=$OUT/${v}_${t}.md
    run_env $v python v28_streamk_owner/check_streamk.py --tile $t --fixups owner --exchanges wt $PARTIALS \
      --jobs 24 > $f 2>&1
    echo "$(date +%H:%M:%S) $v $t $SCOPE: $(tail -1 $f) ($(grep -c '^| [0-9]' $f) configurations)" | tee -a $OUT/summary.txt
  done
  for t in 128x128x128 256x256x64; do
    f=$OUT/${v}_${t}_nofull.md
    run_env $v python v28_streamk_owner/check_streamk.py --tile $t --fixups owner --exchanges wt $PARTIALS \
      --no-full --jobs 24 > $f 2>&1
    echo "$(date +%H:%M:%S) $v $t no-full $SCOPE: $(tail -1 $f) ($(grep -c '^| [0-9]' $f) configurations)" | tee -a $OUT/summary.txt
  done
done
