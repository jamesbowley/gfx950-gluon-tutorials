"""hipBLASLt (via torch, as AITER's `torch` libtype calls it) on lixun_aiter_losses.csv.

    python hbl_env_compare.py <label> <out.csv> [rounds]

Per shape: the kernel hipBLASLt picks (torch profiler), max error vs an fp32 reference,
and TFLOPS from do_bench_cudagraph (median of `rounds`). No AITER import, so it runs in
any torch environment.
"""

import csv
import ctypes
import os
import re
import statistics
import sys
import warnings

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

try:
    from triton.testing import do_bench_cudagraph
except Exception:  # noqa: BLE001
    do_bench_cudagraph = None

SHAPES_CSV = "/home/jbowley/repos/gfx950-gluon-tutorials/experiments/aiter_256x256_loss_investigation/lixun_aiter_losses.csv"


def bench(fn, rep_ms=20):
    if do_bench_cudagraph is not None:
        return do_bench_cudagraph(fn, rep=rep_ms)
    # Same idea as triton's do_bench_cudagraph: time a graph of back-to-back calls.
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        fn()
    end.record()
    torch.cuda.synchronize()
    n = max(1, int(rep_ms / (start.elapsed_time(end) / 5)))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(10):
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        ts.append(start.elapsed_time(end) / n)
    return statistics.mean(ts)


def hipblaslt_info():
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    (x @ x).sum().item()
    path = next(
        l.split()[-1]
        for l in open(f"/proc/{os.getpid()}/maps")
        if "libhipblaslt.so" in l
    )
    lib = ctypes.CDLL(path)
    h, v, buf = ctypes.c_void_p(), ctypes.c_int(), ctypes.create_string_buffer(256)
    lib.hipblasLtCreate(ctypes.byref(h))
    lib.hipblasLtGetVersion(h, ctypes.byref(v))
    lib.hipblasLtGetGitRevision(h, buf)
    return path, v.value, buf.value.decode()


def tile(name):
    mt = re.search(r"MT(\d+x\d+x\d+)", name)
    sk = re.search(r"_SK(\d)", name)
    return f"MT{mt.group(1)}_SK{sk.group(1) if sk else '-'}" if mt else name[:40]


def main():
    label, out_path = sys.argv[1], sys.argv[2]
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    path, ver, rev = hipblaslt_info()
    print(f"[{label}] torch {torch.__version__} hipBLASLt {ver} ({rev}) {path}", flush=True)
    rows = list(csv.DictReader(open(SHAPES_CSV)))
    out = []
    for r in rows:
        M, N, K = int(r["M"]), int(r["N"]), int(r["K"])
        fp32 = r["out"].strip() == "float32"
        torch.manual_seed(0)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        if fp32:
            fn = lambda: torch.mm(x, w.t(), out_dtype=torch.float32)
        else:
            fn = lambda: F.linear(x, w)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            y = fn().float()
            torch.cuda.synchronize()
        fallback = any("HIPBLAS_STATUS" in str(c.message) for c in caught)
        ref = F.linear(x.float(), w.float())
        if not fp32:
            ref = ref.to(torch.bfloat16).float()
        err = (y - ref).abs().max().item()
        ok = bool(torch.allclose(y, ref, atol=1e-1, rtol=1e-2))
        del y, ref
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        kernels = sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})
        main_k = next((k for k in kernels if k.startswith("Cijk")), kernels[0] if kernels else "")
        ts = [bench(fn) for _ in range(rounds)]
        tf = 2.0 * M * N * K / statistics.median(ts) * 1e-9
        row = dict(
            label=label,
            M=M,
            N=N,
            K=K,
            out=r["out"].strip(),
            tiles=int(r["tiles"]),
            csv_pick=r["aiter_pick"],
            csv_aiter_tflops=float(r["aiter_pick_tflops"]),
            tflops=round(tf, 1),
            tile=tile(main_k),
            n_kernels=len(kernels),
            fallback=fallback,
            ok=ok,
            maxerr=f"{err:.3g}",
            hipblaslt=ver,
            rev=rev,
            kernel=main_k,
        )
        out.append(row)
        print(
            f"  {M}x{N}x{K:<6} {row['tile']:22s} {tf:7.1f} TFLOPS"
            f"{' FALLBACK' if fallback else ''}{'' if ok else ' WRONG'}",
            flush=True,
        )
        del x, w
        torch.cuda.empty_cache()
    with open(out_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)


if __name__ == "__main__":
    main()
