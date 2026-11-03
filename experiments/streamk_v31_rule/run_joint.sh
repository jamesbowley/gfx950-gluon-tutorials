#!/bin/bash
# Joint tile x method sweeps on rocm-smi GPU 1 (HIP 3): every spec and every valid dp / owner / rs on
# the tile-sweep shapes (training) and the held-out AITER shapes (test), two rounds each.
set -euo pipefail
cd /home/jbowley/repos/gfx950-gluon-tutorials
R=experiments/streamk_v31_rule/results
S=experiments/streamk_v31_rule/sweep.py
for set in JOINT_TRAIN JOINT_HELDOUT; do
    name=$(echo $set | tr A-Z a-z)
    python $S list --points $set > $R/$name.points
    python $S warm --points @$R/$name.points --jobs 64 > $R/${name}_warm.log 2>&1
    tail -1 $R/${name}_warm.log
    python $S timing --points @$R/$name.points --round 1 --out $R/${name}_r1.csv
    python $S timing --points @$R/$name.points --round 2 --out $R/${name}_r2.csv
    echo "$set DONE"
done
