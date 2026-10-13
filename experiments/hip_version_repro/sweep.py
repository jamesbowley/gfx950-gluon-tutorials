#!/usr/bin/env python3
"""Gluon v9 + llirSched vs AITER's live tuned_gemm pick on the issue's 630 shapes.

    run_<env>.sh python sweep.py <env> [--rounds 2] [--shapes N]
    python derive.py details_<env>.csv supported_status_scoped_<env>.csv

The issue's method, per shape from supported_status_scoped_issue.csv (x (M, K) and
w (N, K) bf16, bias and output dtype from the CSV):

- pick: aiter.tuned_gemm.gemm_a16w16, whatever AITER dispatches today. Its libtype and
  kernel name come from get_GEMM_A16W16_config; the kernels it launches are recorded with
  the torch profiler (for libtype torch these are the hipBLASLt kernels, the CSV's hbl).
- gluon: aiter gemm_a16w16(backend="gluon", kernel_type="compute_bound"), i.e. v9.

Both outputs are checked against an fp32 F.linear reference cast to the output dtype.
Timing is triton.testing.do_bench_cudagraph, `--rounds` interleaved rounds, median. Rows
are appended to details_<env>.csv as each shape finishes and finished shapes are skipped,
so the run can be resumed.
"""

import argparse
import csv
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SHAPES_CSV = os.path.join(HERE, "supported_status_scoped_issue.csv")


def load_shapes():
    with open(SHAPES_CSV, newline="") as f:
        return [
            dict(M=int(r["M"]), N=int(r["N"]), K=int(r["K"]), bias=r["bias"], out=r["out"])
            for r in csv.DictReader(f)
        ]


def key(s):
    return (s["M"], s["N"], s["K"], s["out"])


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {(int(r["M"]), int(r["N"]), int(r["K"]), r["out"]) for r in csv.DictReader(f)}


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def env_info():
    """Versions, and the checks every measurement depends on: GPU 0, llirSched, OPUS."""
    import ctypes

    import torch
    import triton

    import aiter.tuned_gemm as tg
    from aiter.ops.triton import llir_sched

    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0", "must run on GPU 0 only"
    assert torch.cuda.device_count() == 1, torch.cuda.device_count()
    so = llir_sched.plugin_path()
    assert so and llir_sched.compile_options(), "llirSched plugin not available"
    assert tg._opus_launch is not None, "OPUS unavailable: opus rows would silently run torch"
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    (x @ x).sum().item()
    path = next(
        l.split()[-1] for l in open(f"/proc/{os.getpid()}/maps") if "libhipblaslt.so" in l
    )
    lib = ctypes.CDLL(path)
    h, v, buf = ctypes.c_void_p(), ctypes.c_int(), ctypes.create_string_buffer(256)
    lib.hipblasLtCreate(ctypes.byref(h))
    lib.hipblasLtGetVersion(h, ctypes.byref(v))
    lib.hipblasLtGetGitRevision(h, buf)
    p = torch.cuda.get_device_properties(0)
    return dict(
        torch=torch.__version__,
        hip=torch.version.hip,
        hipblaslt=f"{v.value} ({buf.value.decode()})",
        triton=f"{triton.__version__} @ {os.path.dirname(triton.__file__)}",
        llir_sched=so,
        gpu_pci_bus=getattr(p, "pci_bus_id", "n/a"),
        gpu=p.gcnArchName,
    )


def launched_kernels(fn):
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = []
    for e in sorted(
        (e for e in prof.events() if e.device_type.name == "CUDA"),
        key=lambda e: e.time_range.start,
    ):
        if e.name not in names:
            names.append(e.name)
    return names


def run_shape(s, rounds, tuned_keys):
    import torch
    import torch.nn.functional as F
    import triton

    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16 as gluon_gemm
    from aiter.tuned_gemm import gemm_a16w16 as tuned_gemm
    from aiter.tuned_gemm import get_GEMM_A16W16_config

    M, N, K = s["M"], s["N"], s["K"]
    otype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[s["out"]]
    use_bias = s["bias"].strip().lower() == "true"
    torch.manual_seed(0)
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((N,), dtype=torch.bfloat16, device="cuda") if use_bias else None

    cfg_args = dict(bias=use_bias, dtype=str(x.dtype), otype=str(otype))
    cfg = get_GEMM_A16W16_config(M=M, N=N, K=K, scaleAB=False, bpreshuffle=False, **cfg_args)
    libtype = cfg["libtype"]
    kernel = cfg.get("kernelName")
    kernel = "" if kernel is None or kernel != kernel else str(kernel)
    src = "tuned" if (M, N, K, use_bias, str(x.dtype), str(otype)) in tuned_keys else "untuned"

    y = torch.empty((M, N), dtype=otype, device="cuda")
    fns = {
        "pick": lambda: tuned_gemm(x, w, bias, otype),
        "gluon": lambda: gluon_gemm(x, w, bias, otype, y, backend="gluon", kernel_type="compute_bound"),
    }

    ref = F.linear(x.float(), w.float(), None if bias is None else bias.float())
    ref = ref.to(otype).float()
    err, ok, note = {}, {}, []
    for name, fn in fns.items():
        try:
            out = fn().float()
            torch.cuda.synchronize()
            err[name] = (out - ref).abs().max().item()
            ok[name] = bool(torch.allclose(out, ref, atol=1e-1, rtol=1e-2))
            del out
        except Exception as e:  # noqa: BLE001 - record and keep going
            note.append(f"{name}: {type(e).__name__}: {e}"[:300])
            err[name], ok[name] = float("nan"), False
    del ref
    torch.cuda.empty_cache()

    pick_kernels = launched_kernels(fns["pick"]) if err["pick"] == err["pick"] else []

    times = {name: [] for name in fns}
    for _ in range(rounds):
        for name, fn in fns.items():
            if err[name] == err[name]:
                times[name].append(triton.testing.do_bench_cudagraph(fn))
    flops = 2.0 * M * N * K
    tf = {n: (flops / statistics.median(t) * 1e-9 if t else float("nan")) for n, t in times.items()}

    row = dict(
        M=M, N=N, K=K, bias=s["bias"], out=s["out"], src=src,
        gluon_tflops=round(tf["gluon"], 1),
        aiter_pick_tflops=round(tf["pick"], 1),
        aiter_pick=libtype,
        aiter_kernel=kernel,
        hbl_kernels=" | ".join(k for k in pick_kernels if "Cijk_" in k)
        if libtype in ("torch", "hipblaslt")
        else "",
        pick_kernels_launched=" | ".join(pick_kernels),
        gluon_ok=ok["gluon"], pick_ok=ok["pick"],
        gluon_maxerr=f"{err['gluon']:.3g}", pick_maxerr=f"{err['pick']:.3g}",
        gluon_ms=" ".join(f"{t:.4f}" for t in times["gluon"]),
        pick_ms=" ".join(f"{t:.4f}" for t in times["pick"]),
        note="; ".join(note),
    )
    del x, w, y, bias
    torch.cuda.empty_cache()
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("env")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--shapes", type=int, default=None, help="only the first N shapes")
    p.add_argument("--only", default=None, help="comma-separated 0-based shape indices")
    p.add_argument("--out", default=None, help="details CSV (default details_<env>.csv)")
    args = p.parse_args()
    assert args.env == os.environ.get("HVR_ENV"), (args.env, os.environ.get("HVR_ENV"))

    from aiter.tuned_gemm import get_GEMM_A16W16_config_
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx

    info = env_info()
    print(" ".join(f"{k}={v}" for k, v in info.items()), flush=True)
    with open(os.path.join(HERE, f"envinfo_{args.env}.txt"), "w") as f:
        f.writelines(f"{k}={v}\n" for k, v in info.items())

    gfx, cu = get_gfx(), get_cu_num()
    tuned_keys = {
        (k[2], k[3], k[4], bool(k[5]), k[6], k[7])
        for k in get_GEMM_A16W16_config_()
        if k[0] == gfx and k[1] == cu and not k[8] and not k[9]
    }

    out = args.out or os.path.join(HERE, f"details_{args.env}.csv")
    done = done_keys(out)
    shapes = load_shapes()[: args.shapes]
    only = {int(i) for i in args.only.split(",")} if args.only else None
    t0 = time.time()
    for i, s in enumerate(shapes):
        if key(s) in done or (only is not None and i not in only):
            continue
        row = run_shape(s, args.rounds, tuned_keys)
        append_row(out, row)
        print(
            f"[{i + 1}/{len(shapes)} {time.time() - t0:6.0f}s] {s['M']}x{s['N']}x{s['K']} {s['out']:8s} "
            f"pick={row['aiter_pick']:6s} {row['aiter_pick_tflops']:7.1f} gluon={row['gluon_tflops']:7.1f}"
            f"{'' if row['pick_ok'] else ' PICK-WRONG'}{'' if row['gluon_ok'] else ' GLUON-WRONG'}"
            f"{' ' + row['note'] if row['note'] else ''}",
            flush=True,
        )
    print("DONE", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
