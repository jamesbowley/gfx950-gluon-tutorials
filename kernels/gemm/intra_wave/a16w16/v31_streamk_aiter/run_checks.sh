#!/bin/bash
set -euo pipefail

# Quadrant-only v31 coverage. Override TILES, LAUNCHES, or GPU as needed.
source "$(dirname "$0")/env.sh"
TILES=${TILES:-"64x64x64 128x128x128 256x128x64 128x256x64 256x256x64"}
LAUNCHES=${LAUNCHES:-3}

for tile in $TILES; do
    python v31_streamk_aiter/check_streamk.py \
        --tile "$tile" --launches "$LAUNCHES"
    python v31_streamk_aiter/check_streamk.py --tile "$tile" --graph
done

# Stale L2 reader rows must let a reader pass early and read poisoned
# partials. 128x128x128 makes the one-pair skew longer than the reader's
# own store and drain; at 64x64x64 the race is too short to observe.
python v31_streamk_aiter/check_streamk.py --tile 128x128x128 --stale-flag

# A stale barrier count must produce a wrong result (status 0 from the
# negative-control script) or hang (timeout status 124).
set +e
timeout 120 python v31_streamk_aiter/check_streamk.py \
    --tile 64x64x64 --stale-count
status=$?
set -e
case $status in
    0) echo "stale-count produced a wrong result as expected" ;;
    124) echo "stale-count hung as expected" ;;
    *)
        echo "stale-count harness failed with status $status" >&2
        exit "$status"
        ;;
esac
