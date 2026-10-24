"""Synchronization instructions in v30's reduce-scatter builds.

    python v30_streamk_streamlined/asm_check.py --tile 256x256x64

The l2 build must have no atomics and no cache maintenance, and its flag
poll and partial reads must miss L1 (``sc0 nt``). The wt build is the
reference: relaxed counter atomics, write-through partials and ``sc0 sc1``
reads.
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import bench_ext as bench  # noqa: E402
import torch  # noqa: E402
from v30_streamk_streamlined import matmul_kernel  # noqa: E402
from v30_streamk_streamlined.hardware import query_device_config  # noqa: E402
from v30_streamk_streamlined.heuristics import select_plan  # noqa: E402
from v30_streamk_streamlined.selection import TileSpec  # noqa: E402

PATTERNS = {
    "atomics": r"^\s*(global|buffer|flat)_atomic",
    "buffer_wbl2": r"^\s*buffer_wbl2",
    "buffer_inv": r"^\s*buffer_inv",
    "loads sc0 nt": r"^\s*(global|buffer)_load\S*.*\bsc0 nt\b",
    "loads sc0 sc1": r"^\s*(global|buffer)_load\S*.*\bsc0 sc1\b",
    "stores sc0 sc1": r"^\s*(global|buffer)_store\S*.*\bsc0 sc1\b",
}


class Capture:
    def __init__(self, kernel):
        self.kernel = kernel
        self.compiled = None

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            self.compiled = launch(*args, **kwargs)
            return self.compiled

        return run


def build(tile, device, exchange):
    M, N, K = 17 * tile.block_m, 16 * tile.block_n, 8192
    config = select_plan(
        M,
        N,
        K,
        device=device,
        tile=tile,
        policy="one_tile",
        split_n=8,
        fixup="rs",
        exchange=exchange,
    )
    a = torch.rand((M, K), device=bench.DEVICE, dtype=torch.bfloat16) - 0.5
    weights = torch.rand((N, K), device=bench.DEVICE, dtype=torch.bfloat16) - 0.5
    capture = Capture(matmul_kernel._gemm_a16w16_compute_bound_kernel)
    matmul_kernel._gemm_a16w16_compute_bound_kernel = capture
    try:
        matmul_kernel.gemm_a16w16(a, weights, config=config)
    finally:
        matmul_kernel._gemm_a16w16_compute_bound_kernel = capture.kernel
    torch.cuda.synchronize()
    return capture.compiled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile", default="256x256x64")
    args = parser.parse_args()
    tile = TileSpec(*(int(part) for part in args.tile.split("x")))
    device = query_device_config()
    print(f"# v30 reduce-scatter synchronization, {args.tile}\n")
    print("| exchange | VGPRs | spills | " + " | ".join(PATTERNS) + " |")
    print("|---" * (len(PATTERNS) + 3) + "|")
    failures = []
    for exchange in ("wt", "l2"):
        compiled = build(tile, device, exchange)
        asm = compiled.asm["amdgcn"]
        counts = {
            name: len(re.findall(pattern, asm, re.MULTILINE))
            for name, pattern in PATTERNS.items()
        }
        vgprs = re.search(r"\.vgpr_count:\s+(\d+)", asm)
        spills = re.search(r"\.vgpr_spill_count:\s+(\d+)", asm)
        print(
            f"| {exchange} | {vgprs.group(1) if vgprs else '?'} | "
            f"{spills.group(1) if spills else '?'} | "
            + " | ".join(str(counts[name]) for name in PATTERNS)
            + " |"
        )
        if exchange == "l2":
            for name in ("atomics", "buffer_wbl2", "buffer_inv"):
                if counts[name]:
                    failures.append(f"l2 has {counts[name]} {name}")
            if not counts["loads sc0 nt"]:
                failures.append("l2 has no sc0 nt loads")
    if failures:
        raise AssertionError("; ".join(failures))
    print("\nl2: no atomics, no cache maintenance, sc0 nt loads present")


if __name__ == "__main__":
    main()
