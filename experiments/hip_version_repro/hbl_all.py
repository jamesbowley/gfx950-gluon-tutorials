#!/usr/bin/env python3
"""hipBLASLt on all 630 issue shapes, whatever AITER picks for them.

    run_<env>.sh python hbl_all.py <env>

Per shape: the call AITER's `torch` libtype makes (F.linear, or torch.mm(out_dtype=fp32) for
fp32 output), the hipBLASLt kernels it launches (torch profiler) and TFLOPS from
do_bench_cudagraph, median of 2 rounds. Writes hbl_all_<env>.csv.
"""

import csv
import os
import statistics
import sys

import torch
import torch.nn.functional as F
import triton
from torch.profiler import ProfilerActivity, profile

HERE = os.path.dirname(os.path.abspath(__file__))

assert os.environ.get("HIP_VISIBLE_DEVICES") == "0" and torch.cuda.device_count() == 1
env = sys.argv[1]
assert env == os.environ.get("HVR_ENV")
rows = list(csv.DictReader(open(os.path.join(HERE, "supported_status_scoped_issue.csv"))))
out = []
for i, r in enumerate(rows):
    M, N, K = int(r["M"]), int(r["N"]), int(r["K"])
    fp32 = r["out"] == "float32"
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    if fp32:
        fn = lambda: torch.mm(x, w.t(), out_dtype=torch.float32)  # noqa: E731
    else:
        fn = lambda: F.linear(x, w)  # noqa: E731
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = []
    for e in sorted((e for e in prof.events() if e.device_type.name == "CUDA"), key=lambda e: e.time_range.start):
        if "Cijk_" in e.name and e.name not in names:
            names.append(e.name)
    ts = [triton.testing.do_bench_cudagraph(fn) for _ in range(2)]
    tf = 2.0 * M * N * K / statistics.median(ts) * 1e-9
    out.append(dict(M=M, N=N, K=K, out=r["out"], aiter_pick=r["aiter_pick"], hbl_tflops=round(tf, 1),
                    hbl_kernels=" | ".join(names)))
    print(f"[{i + 1}/{len(rows)}] {M}x{N}x{K} {tf:7.1f}", flush=True)
    del x, w
    torch.cuda.empty_cache()
with open(os.path.join(HERE, f"hbl_all_{env}.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(out[0]))
    wr.writeheader()
    wr.writerows(out)
print("DONE")
