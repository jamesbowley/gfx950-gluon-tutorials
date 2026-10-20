#!/bin/bash
# GPU=<n> bash v27_streamk_tile/run_checks.sh TILE [TILE ...]: check_streamk.py per tile, one after the
# other on one GPU (two stream-K kernels on one GPU need not all be resident), to results/check_streamk_<tile>.md.
source "$(dirname "$0")/env.sh"
for t in "$@"; do
    python v27_streamk_tile/check_streamk.py --tile "$t" --jobs ${JOBS:-12} > "v27_streamk_tile/results/check_streamk_$t.md" \
        2> "v27_streamk_tile/tmp/check_streamk_$t.err"
done
