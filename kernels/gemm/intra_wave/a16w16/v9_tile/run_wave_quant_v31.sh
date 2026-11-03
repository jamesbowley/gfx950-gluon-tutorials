# Wave-quantization sweep for one K in one session: the WQ{K} data-parallel tiles (v9_tile env,
# amdgcnas) and the WQ31_{K} v31_streamk_aiter points (v31 env, no plugins), timing (back to back)
# and clock/power for each, under the GPU guard. GPU is the HIP index; SMI the rocm-smi index of the
# same device (they match on GPU 6, PCI e5:00.0).
#
#     GPU=6 SMI=6 bash v9_tile/run_wave_quant_v31.sh 4096
set -euo pipefail
K=$1
A16=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
R=v9_tile/results
T=v9_tile/tmp/wq31_$K
GUARD="python $A16/../../../../experiments/streamk_v31_rule/gpu_guard.py --smi-index ${SMI:?} --log $A16/$R/wq31_$K.guard --"
(
    source "$A16/v9_tile/env.sh"
    S="python v9_tile/sweep.py --tmp $T --rotating-mb 2048"
    $GUARD $S timing --graph --points WQ$K --out $R/wq31dp${K}_timing_graph.csv
    $GUARD $S clock --points WQ$K --out $R/wq31dp${K}_clock.csv
)
(
    source "$A16/v31_streamk_aiter/env.sh"
    S="python v9_tile/sweep.py --tmp $T --rotating-mb 2048"
    $GUARD $S timing --graph --points WQ31_$K --out $R/wq31_${K}_timing_graph.csv
    $GUARD $S clock --points WQ31_$K --out $R/wq31_${K}_clock.csv
)
