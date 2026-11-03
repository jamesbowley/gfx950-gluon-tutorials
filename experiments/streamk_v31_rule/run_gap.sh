#!/bin/bash
# The gap sweep (S and k between the grid's) on rocm-smi GPU 1 (HIP 3): warm compile, then two timing rounds.
set -euo pipefail
cd /home/jbowley/repos/gfx950-gluon-tutorials
R=experiments/streamk_v31_rule/results
python experiments/streamk_v31_rule/sweep.py warm --points GAP --jobs 64 > $R/gap_warm.log 2>&1
tail -3 $R/gap_warm.log
python experiments/streamk_v31_rule/sweep.py timing --points GAP --round 1 --out $R/gap_r1.csv
python experiments/streamk_v31_rule/sweep.py timing --points GAP --round 2 --out $R/gap_r2.csv
echo "GAP DONE"
