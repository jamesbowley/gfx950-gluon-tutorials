#!/bin/bash
# The held-out AITER shapes on rocm-smi GPU 1 (HIP 3): warm compile, then two timing rounds.
set -euo pipefail
cd /home/jbowley/repos/gfx950-gluon-tutorials
R=experiments/streamk_v31_rule/results
python experiments/streamk_v31_rule/sweep.py warm --points HELDOUT --jobs 64 > $R/heldout_warm.log 2>&1
tail -3 $R/heldout_warm.log
python experiments/streamk_v31_rule/sweep.py timing --points HELDOUT --round 1 --out $R/heldout_r1.csv
python experiments/streamk_v31_rule/sweep.py timing --points HELDOUT --round 2 --out $R/heldout_r2.csv
echo "HELDOUT DONE"
