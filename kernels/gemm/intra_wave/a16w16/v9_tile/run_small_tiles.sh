# Small-tile sweeps (WORKLOG §14) on one GPU, 2 GB of rotating inputs:
#   cb    compute-bound SMALL_CB: timing back to back, clock/power, counters; ATT at K = 8192
#   mb    memory-bound SMALL_MB: timing back to back, counters
#   warm  SMALL_WARM with the inputs warm in the MALL: timing back to back, ATT
#   alt   SMALL_ALT_* (4x64 MFMA, replicated warps, waves_per_eu) and the Triton reference SMALL_REF_*
#
#     GPU=3 bash v9_tile/run_small_tiles.sh cb|mb|warm|alt|all
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
R=v9_tile/results
T=v9_tile/tmp/small
S="python v9_tile/sweep.py --tmp $T --rotating-mb 2048"
what=${1:-all}
if [[ $what == cb || $what == all ]]; then
    CB8=$(python -c "import sys; sys.path.insert(0, 'v9_tile'); import sweep
print(','.join(p for p in sweep.POINTS['SMALL_CB'] if p.endswith('x8192')))")
    $S timing --graph --points SMALL_CB --out $R/small_cb_timing_graph.csv
    $S clock --points SMALL_CB --out $R/small_cb_clock.csv
    $S counters --points SMALL_CB --out $R/small_cb_counters.csv
    $S att --points "$CB8" --out $R/small_cb_att.csv
fi
if [[ $what == mb || $what == all ]]; then
    $S timing --graph --points SMALL_MB --out $R/small_mb_timing_graph.csv
    $S counters --points SMALL_MB --out $R/small_mb_counters.csv
fi
if [[ $what == alt || $what == all ]]; then
    # v9_tile_small's alternatives under the usual stack; the Triton reference stock (no plugins).
    $S timing --graph --points SMALL_ALT_CB --out $R/small_alt_cb_timing_graph.csv
    $S counters --points SMALL_ALT_CB --out $R/small_alt_cb_counters.csv
    $S att --points SMALL_ALT_CB --out $R/small_alt_cb_att.csv
    $S timing --graph --points SMALL_ALT_MB --out $R/small_alt_mb_timing_graph.csv
    (
        unset LLVM_PASS_PLUGIN_PATH TRITON_AMDGCNAS_PLUGIN
        $S timing --graph --points SMALL_REF_CB --out $R/small_ref_cb_timing_graph.csv
        $S counters --points SMALL_REF_CB --out $R/small_ref_cb_counters.csv
        $S att --points SMALL_REF_CB --out $R/small_ref_cb_att.csv
        $S timing --graph --points SMALL_REF_MB --out $R/small_ref_mb_timing_graph.csv
    )
fi
if [[ $what == warm || $what == all ]]; then
    W="python v9_tile/sweep.py --tmp $T --rotating-mb 0"
    $W timing --graph --points SMALL_WARM --out $R/small_warm_timing_graph.csv
    $W att --points SMALL_WARM --out $R/small_warm_att.csv
fi
