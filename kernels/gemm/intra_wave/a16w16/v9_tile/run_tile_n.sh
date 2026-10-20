# 256x256 down to 256x128 in steps of 16 on N (WORKLOG §19), each on its 16x16 grid (sweep.py's TILE_N), cold
# inputs, the MFMA scheduler without amdgcnas (256x192 is miscompiled with it, §18.3):
#
#     GPU=1 bash v9_tile/run_tile_n.sh            # SKIP_CHECKS=1 skips the correctness gate and asm_check
#
# Back-to-back (graph) and isolated timing, hwmon clock and power, counter sets 0 and 1, then ATT three times
# (launch 10 of 12, CU 0 of SE0-3), each trace reduced by att_stalls.py and att_gaps.py.
# python v9_tile/report_tile_n.py merges the results.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
unset TRITON_AMDGCNAS_PLUGIN
TILES=256x256,256x240,256x224,256x208,256x192,256x176,256x160,256x144,256x128
R=v9_tile/results
T=v9_tile/tmp/tile_n
mkdir -p "$R/tile_n" "$T"
S="python v9_tile/sweep.py --points TILE_N --rotating-mb 2048"
if [ -z "${SKIP_CHECKS:-}" ]; then
    python v9_tile/check_tiles.py $TILES > $R/check_tiles_tile_n.md
    python v9_tile/asm_check.py $TILES --grid16 > $R/asm_tile_n.md
fi
rocm-smi -d "$HIP_VISIBLE_DEVICES" --showclocks --showperflevel --showuniqueid > "$R/tile_n/smi_before.txt" 2>&1 || true
$S timing --graph --tmp $T/timing --out $R/tile_n_timing_graph.csv
$S timing --tmp $T/timing --out $R/tile_n_timing.csv
$S clock --tmp $T/clock --out $R/tile_n_clock.csv
$S counters --sets 0,1 --tmp $T/counters --out $R/tile_n_counters.csv
for r in 1 2 3; do
    $S att --tmp $T/att_r$r --out $R/tile_n_att_r$r.csv
    python v9_tile/att_stalls.py $T/att_r$r/sweep_att/* > $R/tile_n/att_stalls_r$r.md
    python v9_tile/att_gaps.py $T/att_r$r/sweep_att/* --json $R/tile_n/att_gaps_r$r.json > $R/tile_n/att_gaps_r$r.md
done
rocm-smi -d "$HIP_VISIBLE_DEVICES" --showclocks --showperflevel > "$R/tile_n/smi_after.txt" 2>&1 || true
python v9_tile/report_tile_n.py > $R/tile_n/report.md
