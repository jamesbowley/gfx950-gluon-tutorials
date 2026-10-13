# Wave-quantization sweep for one K: timing (back to back) and clock/power for the WQ{K} tiles and
# the WQSK{K} stream-K points. The stream-K points run without the amdgcnas hook, as v24-v26 were
# measured. The rotating inputs are 2 GB: at 512 MB the largest points (one copy of A, B and C
# over 256 MB) get a single copy, which stays warm in the MALL.
#
#     GPU=6 bash v9_tile/run_wave_quant_tiles.sh 4096
set -euo pipefail
K=$1
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
R=v9_tile/results
T=v9_tile/tmp/wq$K
S="python v9_tile/sweep.py --tmp $T --rotating-mb 2048"
$S timing --graph --points WQ$K --out $R/wq${K}_timing_graph.csv
$S clock --points WQ$K --out $R/wq${K}_clock.csv
(
    unset TRITON_AMDGCNAS_PLUGIN
    $S timing --graph --points WQSK$K --out $R/wqsk${K}_timing_graph.csv
    $S clock --points WQSK$K --out $R/wqsk${K}_clock.csv
)
