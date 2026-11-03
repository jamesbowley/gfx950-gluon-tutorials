"""experiments/streamk_v31_rule/sweep.py with two extra kinds:
- ``2tg{G}``: two_tile on G programs (reduced_grid.py);
- ``def``: the v31 default plan (a tile-only config).

Same command line as sweep.py; its worker subprocesses run this file too.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "streamk_v31_rule"))
sys.path.insert(0, HERE)
import grid  # noqa: E402
import sweep  # noqa: E402

_config_for, _valid = grid.config_for, grid.valid


def streamk_grid(kind):
    return int(kind[3:]) if kind.startswith("2tg") else None


def config_for(kind, tile):
    if kind.startswith("2tg"):
        return _config_for("2t", tile)
    if kind == "def":
        return dict(_config_for("dp", tile), STREAMK_POLICY="default", STREAMK_FIXUP="", STREAMK_EXCHANGE="")
    return _config_for(kind, tile)


def valid(kind, tile, M, N, K):
    G = streamk_grid(kind)
    if G is not None:
        sh = grid.shape_info(tile, M, N, K)
        return G % grid.NUM_XCDS == 0 and G <= grid.P and G < sh["T"] < 2 * G and sh["pairs"] >= 2
    if kind == "def":
        return True
    return _valid(kind, tile, M, N, K)


def run_fn(kind, tile):
    import reduced_grid

    bm, bn, bk = grid.parse_tile(tile)
    config = dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk) if kind == "def" else config_for(kind, tile)
    G = streamk_grid(kind)

    def run(x, w, y):
        reduced_grid.gemm(x, w, y=y, config=config, G=G)

    return run


grid.config_for, grid.valid = config_for, valid
sweep.run_fn = run_fn
sweep.__file__ = os.path.abspath(__file__)

if __name__ == "__main__":
    sweep.main()
