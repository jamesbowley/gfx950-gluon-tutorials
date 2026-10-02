"""Kernel names hipBLASLt picks per CSV shape, from TENSILE_DB logging (set by the caller).

    TENSILE_DB=0x8000 python hbl_kernel_names.py > log
"""

import csv
import sys

import torch
import torch.nn.functional as F

SHAPES_CSV = "/home/jbowley/repos/gfx950-gluon-tutorials/experiments/aiter_256x256_loss_investigation/lixun_aiter_losses.csv"

for r in csv.DictReader(open(SHAPES_CSV)):
    M, N, K = int(r["M"]), int(r["N"]), int(r["K"])
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    print(f"@@SHAPE {M}x{N}x{K}", flush=True)
    if r["out"].strip() == "float32":
        torch.mm(x, w.t(), out_dtype=torch.float32)
    else:
        F.linear(x, w)
    torch.cuda.synchronize()
    sys.stdout.flush()
    del x, w
    torch.cuda.empty_cache()
