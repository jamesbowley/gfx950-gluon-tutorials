# One wave of 256x256 for one K in one session (WORKLOG §20), under the GPU guard: SQ (near-square
# grids, 16..256 tiles), SHAPEGRID (square to 1 x T at T busy CUs), SKIDEAL (the full chip at the
# same total k-steps) and the figure's first wave again (WQ1W data-parallel, WQ31_1W v31) for the
# overlay check. Timing (back to back) and clock/power for each, counters for the HBM traffic, and a
# warm clock pass (inputs in the MALL) of the SHAPEGRID points that fit. GPU is the HIP index, SMI
# the rocm-smi index of the same device (they match on GPU 6, PCI e5:00.0).
#
#     GPU=6 SMI=6 bash v9_tile/run_one_wave.sh 4096
set -euo pipefail
K=$1
A16=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
R=v9_tile/results
T=v9_tile/tmp/ow_$K
GUARD="python $A16/../../../../experiments/streamk_v31_rule/gpu_guard.py --smi-index ${SMI:?} --log $A16/$R/ow_$K.guard --"
(
    source "$A16/v9_tile/env.sh"
    S="python v9_tile/sweep.py --tmp $T --rotating-mb 2048"
    for P in WQ1W SQ SHAPEGRID SKIDEAL; do
        p=$(echo $P | tr A-Z a-z)
        $GUARD $S timing --graph --points $P$K --out $R/ow_${p}${K}_timing_graph.csv
        $GUARD $S clock --points $P$K --out $R/ow_${p}${K}_clock.csv
    done
    for P in WQ1W SQ SHAPEGRID SKIDEAL; do
        p=$(echo $P | tr A-Z a-z)
        $GUARD $S counters --sets 0,1,2 --points $P$K --out $R/ow_${p}${K}_counters.csv
    done
    $GUARD python v9_tile/sweep.py --tmp $T --rotating-mb 0 clock --points SHAPEGRID_WARM$K \
        --out $R/ow_shapegrid${K}_warm_clock.csv
)
(
    source "$A16/v31_streamk_aiter/env.sh"
    S="python v9_tile/sweep.py --tmp $T --rotating-mb 2048"
    $GUARD $S timing --graph --points WQ31_1W$K --out $R/ow_wq31${K}_timing_graph.csv
    $GUARD $S clock --points WQ31_1W$K --out $R/ow_wq31${K}_clock.csv
)
