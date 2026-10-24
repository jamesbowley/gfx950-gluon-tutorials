#!/usr/bin/env python3
"""origami_ranks.json: origami's tile ranking for every shape of the exhaustive v27 sweeps, over the tiles each shape
was swept at (make_origami_picks.Origami's configs), for prune_eval.py --origami.

    source $A16/v28_streamk_owner/env.sh; export HIP_VISIBLE_DEVICES=3
    PYTHONPATH=/home/jbowley/origami_build/site:$PYTHONPATH python $AIS/origami_ranks.py
"""

import collections
import csv
import json
import os

from make_origami_picks import Origami
from prune_eval import DATASETS, RES

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    o = Origami()
    tiles = collections.defaultdict(set)
    for f in DATASETS:
        for r in csv.DictReader(open(os.path.join(RES, f))):
            tiles[(int(r["M"]), int(r["N"]), int(r["K"]))].add(r["tile"])
    out = {f"{M}x{N}x{K}": [t for t, _ in o.rank(M, N, K, sorted(ts))] for (M, N, K), ts in tiles.items()}
    path = os.path.join(HERE, "origami_ranks.json")
    json.dump(out, open(path, "w"))
    print(f"{len(out)} shapes -> {path}")


if __name__ == "__main__":
    main()
