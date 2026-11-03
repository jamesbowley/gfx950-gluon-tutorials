#!/bin/bash
# The v31 rule sweep on rocm-smi GPU 1 (HIP 3): warm compile, then two timing rounds.
set -euo pipefail
cd /home/jbowley/repos/gfx950-gluon-tutorials
R=experiments/streamk_v31_rule/results
python experiments/streamk_v31_rule/sweep.py warm --points GRID --jobs 64 > $R/grid_warm.log 2>&1
tail -3 $R/grid_warm.log
python experiments/streamk_v31_rule/sweep.py timing --points GRID --round 1 --out $R/grid_r1.csv
python experiments/streamk_v31_rule/sweep.py timing --points GRID --round 2 --out $R/grid_r2.csv
echo "GRID DONE"
