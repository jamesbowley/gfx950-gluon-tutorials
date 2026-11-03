#!/bin/bash
# The tile-choice training sweep on rocm-smi GPU 1 (HIP 3): warm compile, then two timing rounds.
set -euo pipefail
cd /home/jbowley/repos/gfx950-gluon-tutorials
R=experiments/streamk_v31_rule/results
python experiments/streamk_v31_rule/sweep.py warm --points TILESWEEP --jobs 64 > $R/tilesweep_warm.log 2>&1
tail -3 $R/tilesweep_warm.log
python experiments/streamk_v31_rule/sweep.py timing --points TILESWEEP --round 1 --out $R/tilesweep_r1.csv
python experiments/streamk_v31_rule/sweep.py timing --points TILESWEEP --round 2 --out $R/tilesweep_r2.csv
echo "TILESWEEP DONE"
