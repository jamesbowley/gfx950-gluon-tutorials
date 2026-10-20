# Per-tile load latency and MFMA efficiency (WORKLOG §16) on one GPU, 2 GB of rotating inputs, the
# v3.0 MFMA scheduler alone (amdgcnas off):
#   calib  calib_latency.py: the latency counters on experiments/tcp_latency's probe and stream kernels
#   fw     LAT_FW (one full wave per tile, K = 8192): timing back to back, clock/power, counters, ATT
#   cb     LAT_CB (2048x4096x8192 for every tile): the same
#
#     GPU=3 bash v9_tile/run_latency.sh calib|fw|cb|all
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
unset TRITON_AMDGCNAS_PLUGIN
R=v9_tile/results
what=${1:-all}
if [[ $what == calib || $what == all ]]; then
    python v9_tile/calib_latency.py --out $R/latency_calib.csv
fi
for p in fw cb; do
    if [[ $what == "$p" || $what == all ]]; then
        # One scratch directory per preset, so fw and cb can run at once on two GPUs.
        S="python v9_tile/sweep.py --tmp v9_tile/tmp/latency/$p --rotating-mb 2048"
        P=LAT_${p^^}
        $S timing --graph --points $P --out $R/lat_${p}_timing_graph.csv
        $S clock --points $P --out $R/lat_${p}_clock.csv
        $S counters --sets 0,1,3,5,6,7 --points $P --out $R/lat_${p}_counters.csv
        $S att --points $P --out $R/lat_${p}_att.csv
    fi
done
