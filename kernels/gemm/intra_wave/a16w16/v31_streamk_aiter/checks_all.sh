#!/bin/bash
# Every v31 check: run_checks.sh, the stream and persistent-partial modes, the extra cases.
set -euo pipefail
source "$(dirname "$0")/env.sh"
python -m unittest v31_streamk_aiter.test_selection
bash v31_streamk_aiter/run_checks.sh
for tile in 64x64x64 128x128x128 256x256x64; do
    python v31_streamk_aiter/check_streamk.py --tile "$tile" --streams
    python v31_streamk_aiter/check_streamk.py --tile "$tile" --persistent-partials
    python v31_streamk_aiter/check_streamk.py --tile "$tile" --extra
done
echo "ALL CHECKS PASSED"
