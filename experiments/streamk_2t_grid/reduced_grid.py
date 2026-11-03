"""Two-tile on a reduced grid of G programs, without changing v31.

v31's end-to-end path splits over NUM_PROGRAMS, which gemm_a16w16 takes from
streamk_topology(). Reporting G programs instead runs two_tile on G programs; with
G < T < 2 G every tile is in the end-to-end split and no program strides a
data-parallel loop over the smaller grid. G must be a multiple of the XCD count and
at most the CUs.
"""

import contextlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "kernels", "gemm", "intra_wave", "a16w16")))
from v31_streamk_aiter import matmul_kernel as mk  # noqa: E402


def check(M, N, config, G):
    num_programs, num_xcds = mk.streamk_topology()
    if config["STREAMK_POLICY"] != "two_tile":
        raise ValueError(f"a reduced grid requires two_tile, got {config['STREAMK_POLICY']}")
    if type(G) is not int or G <= 0 or G % num_xcds or G > num_programs:
        raise ValueError(f"G={G} must be a positive multiple of {num_xcds} and at most {num_programs}")
    T = mk.triton.cdiv(M, config["BLOCK_M"]) * mk.triton.cdiv(N, config["BLOCK_N"])
    if not G < T < 2 * G:
        raise ValueError(f"G={G} needs G < T < 2 G, got T={T}")


@contextlib.contextmanager
def topology(G):
    real = mk.streamk_topology
    num_xcds = real()[1]
    mk.streamk_topology = lambda: (G, num_xcds)
    try:
        yield
    finally:
        mk.streamk_topology = real


def gemm(x, w, bias=None, y=None, config=None, G=None, persistent_partials=False):
    """v31's gemm_a16w16 with two_tile on G programs (G None: unchanged)."""
    if G is None:
        return mk.gemm_a16w16(x, w, bias, dtype=x.dtype, y=y, config=config,
                              persistent_partials=persistent_partials)
    check(x.shape[0], w.shape[0], config, G)
    with topology(G):
        return mk.gemm_a16w16(x, w, bias, dtype=x.dtype, y=y, config=config,
                              persistent_partials=persistent_partials)
