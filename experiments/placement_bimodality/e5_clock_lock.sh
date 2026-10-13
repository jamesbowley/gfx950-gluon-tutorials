#!/bin/bash
# E5: rerun E1 and a coarse E2 sweep with GPU 2's gfx clock capped (performance determinism).
#
#     bash e5_clock_lock.sh [SCLK_MHZ]       (default 1800; needs root; resets on exit)
#
# The cap is below every clock seen back-to-back (1920-2050 MHz), so both halves should run at
# it. HIP_VISIBLE_DEVICES=2 and amd-smi GPU 2 are the same device on this machine.
set -euo pipefail
SCLK=${1:-1800}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/results/e5_raw_${SCLK}.txt

reset() {
    amd-smi reset -g 2 -d >/dev/null 2>&1 || true
    echo "reset perf determinism on GPU 2" >&2
}
trap reset EXIT

amd-smi set -g 2 -d "$SCLK"
sleep 2
{
    echo "== sclk cap $SCLK MHz"
    for s in 0 1; do
        echo "== e1 seed $s"
        python "$HERE/place.py" e1 --n 10 --seed $s 2>&1 | grep -E "^\||classified|warning"
    done
    echo "== sweep C 0-512 MB step 8, A/B at 900 MB"
    python "$HERE/place.py" sweep --what c --start 0 --stop 512 --step 8 --fixed 900 2>&1 | grep -E "^\||base"
} | tee "$OUT"
