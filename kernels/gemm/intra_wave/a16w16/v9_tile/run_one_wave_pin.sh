# The SHAPEGRID clock/power pass at a pinned shader clock (WORKLOG §20): with the clock fixed, the
# power differences across shapes at the same busy-CU count are the memory traffic's, and across
# busy-CU counts the CUs'. Pin from the host first (the container cannot write sysfs):
#     sudo rocm-smi -d 6 --setperfdeterminism 1500
# then
#     GPU=6 SMI=6 bash v9_tile/run_one_wave_pin.sh 4096 1500
# and afterwards `sudo rocm-smi -d 6 --resetperfdeterminism`.
set -euo pipefail
K=$1
MHZ=$2
A16=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
R=v9_tile/results
T=v9_tile/tmp/ow_${K}_pin$MHZ
GUARD="python $A16/../../../../experiments/streamk_v31_rule/gpu_guard.py --smi-index ${SMI:?} --log $A16/$R/ow_$K.guard --"
source "$A16/v9_tile/env.sh"
rocm-smi -d "$SMI" --showclocks --showperflevel > "$R/ow_shapegrid${K}_pin${MHZ}_smi.txt" 2>&1
$GUARD python v9_tile/sweep.py --tmp $T --rotating-mb 2048 clock --points SHAPEGRID$K \
    --out $R/ow_shapegrid${K}_pin${MHZ}_clock.csv
rocm-smi -d "$SMI" --showclocks --showperflevel >> "$R/ow_shapegrid${K}_pin${MHZ}_smi.txt" 2>&1
