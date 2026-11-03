#!/bin/bash
# The default plan's checks: tile-only configs at three tiles (BLOCK_K 64 large, BLOCK_K 64 with a side
# of 64, BLOCK_K 128), then graph replay and the stream check unchanged.
set -euo pipefail
source "$(dirname "$0")/env.sh"
python -m unittest v31_streamk_aiter.test_selection
for tile in 256x256x64 128x64x64 128x128x128; do
    python v31_streamk_aiter/check_streamk.py --tile "$tile" --default
done
echo "DEFAULT CHECKS PASSED"
