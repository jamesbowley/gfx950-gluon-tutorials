"""The kernel AITER's tuned_gemm.gemm_a16w16 launches for each shape in lixun_aiter_losses.csv.

    TENSILE_DB=0x8000 HIP_VISIBLE_DEVICES=0 run_aiter_compare_hbl122.sh-style env \
        python aiter_pick_kernels.py > aiter_pick_kernels_hbl122.log

Prints "@@SHAPE M N K libtype kernelName" before the timed-path call (a warm-up call runs
under "@@WARMUP" first); for libtype torch, hipBLASLt's TENSILE_DB logging then prints the
"Running kernel:" line.
"""

import csv
import os
import sys

import torch

from aiter.tuned_gemm import gemm_a16w16, get_GEMM_A16W16_config

HERE = os.path.dirname(os.path.abspath(__file__))
SHAPES_CSV = os.path.join(HERE, "..", "..", "lixun_aiter_losses.csv")

for r in csv.DictReader(open(SHAPES_CSV)):
    M, N, K = int(r["M"]), int(r["N"]), int(r["K"])
    otype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[r["out"].strip()]
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    cfg = get_GEMM_A16W16_config(
        M=M, N=N, K=K, bias=False, dtype=str(x.dtype), otype=str(otype),
        scaleAB=False, bpreshuffle=False,
    )
    print("@@WARMUP", flush=True)
    gemm_a16w16(x, w, None, otype)  # first call: JIT builds and config lookup
    torch.cuda.synchronize()
    sys.stdout.flush()
    print(f"@@SHAPE {M} {N} {K} {cfg['libtype']} {cfg.get('kernelName') or '-'}", flush=True)
    gemm_a16w16(x, w, None, otype)
    torch.cuda.synchronize()
    sys.stdout.flush()
    del x, w
    torch.cuda.empty_cache()
