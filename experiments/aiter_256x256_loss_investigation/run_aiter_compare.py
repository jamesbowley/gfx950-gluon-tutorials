#!/usr/bin/env python3
"""AITER's live tuned_gemm pick vs the AITER Gluon a16w16 kernels on lixun_aiter_losses.csv.

    HIP_VISIBLE_DEVICES=0 \
    PYTHONPATH=<deepbind shim>:<triton gfx950-tutorial-v2.2>/python:<aiter> \
    python run_aiter_compare.py [--shapes N] [--rounds 3]
    python run_aiter_compare.py --summarize
    results/hipblaslt_versions/run_aiter_compare_hbl122.sh   # same, on hipBLASLt 1.2.2

Per shape, in one process and on the same inputs (x (M, K), w (N, K) bf16, bias and output
dtype from the CSV):

- pick: aiter.tuned_gemm.gemm_a16w16, i.e. whatever AITER dispatches today; its libtype and
  kernel name come from get_GEMM_A16W16_config.
- v9, v10..v13: aiter gemm_a16w16(backend="gluon", kernel_type=compute_bound[_v1X]).

Every output is checked against an fp32 F.linear reference cast to the output dtype. Timing
is triton.testing.do_bench_cudagraph, `--rounds` interleaved rounds over all variants, median
(the issue's method). Rows are appended to results/aiter_live_compare.csv as each shape
finishes, and shapes already there are skipped, so the run can be resumed.
"""

import argparse
import csv
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
SHAPES_CSV = os.path.join(HERE, "lixun_aiter_losses.csv")
RESULTS = os.path.join(HERE, "results")
OUT_CSV = os.path.join(RESULTS, "aiter_live_compare.csv")
SUMMARY = os.path.join(RESULTS, "aiter_live_compare.md")
GLUON = {
    "v9": "compute_bound",
    "v10": "compute_bound_v10",
    "v11": "compute_bound_v11",
    "v12": "compute_bound_v12",
    "v13": "compute_bound_v13",
}
VARIANTS = ["pick", *GLUON]
CUS = 256


def load_shapes():
    with open(SHAPES_CSV, newline="") as f:
        return [
            dict(
                M=int(r["M"]),
                N=int(r["N"]),
                K=int(r["K"]),
                bias=r["bias"].strip().upper() == "TRUE",
                out=r["out"].strip(),
                tiles=int(r["tiles"]),
                csv_gluon=float(r["gluon_tflops"]),
                csv_aiter=float(r["aiter_pick_tflops"]),
                csv_pick=r["aiter_pick"],
                csv_kernel=r["aiter_kernel"],
                csv_gap=float(r["gap_pct"]),
            )
            for r in csv.DictReader(f)
        ]


def bucket(tiles):
    if tiles < CUS:
        return "few tiles (< 1 wave)"
    if tiles % CUS:
        return "partial waves"
    return "full waves"


def key(s):
    return (s["M"], s["N"], s["K"])


def read_rows():
    if not os.path.exists(OUT_CSV):
        return []
    with open(OUT_CSV, newline="") as f:
        return list(csv.DictReader(f))


def append_row(row):
    new = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def run_shape(s, rounds):
    import torch
    import torch.nn.functional as F
    import triton

    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16 as gluon_gemm
    from aiter.tuned_gemm import gemm_a16w16 as tuned_gemm
    from aiter.tuned_gemm import get_GEMM_A16W16_config

    M, N, K = key(s)
    otype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[s["out"]]
    torch.manual_seed(0)
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((N,), dtype=torch.bfloat16, device="cuda") if s["bias"] else None

    cfg = get_GEMM_A16W16_config(
        M=M,
        N=N,
        K=K,
        bias=bias is not None,
        dtype=str(x.dtype),
        otype=str(otype),
        scaleAB=False,
        bpreshuffle=False,
    )
    libtype = cfg["libtype"]
    kernel = cfg.get("kernelName") or ""

    y = torch.empty((M, N), dtype=otype, device="cuda")
    fns = {"pick": lambda: tuned_gemm(x, w, bias, otype)}
    for name, kind in GLUON.items():
        fns[name] = lambda kind=kind: gluon_gemm(
            x, w, bias, otype, y, backend="gluon", kernel_type=kind
        )

    ref = F.linear(x.float(), w.float(), None if bias is None else bias.float())
    ref = ref.to(otype).float()
    errors, ok = {}, {}
    for name, fn in fns.items():
        try:
            out = fn().float()
            torch.cuda.synchronize()
            errors[name] = (out - ref).abs().max().item()
            ok[name] = bool(torch.allclose(out, ref, atol=1e-1, rtol=1e-2))
            del out
        except Exception as e:  # noqa: BLE001 - record and keep going
            print(f"    {name}: {type(e).__name__}: {e}", flush=True)
            errors[name], ok[name] = float("nan"), False
    del ref
    torch.cuda.empty_cache()

    times = {name: [] for name in fns}
    for _ in range(rounds):
        for name, fn in fns.items():
            if errors[name] != errors[name]:  # nan: it failed to run
                continue
            times[name].append(triton.testing.do_bench_cudagraph(fn))
    flops = 2.0 * M * N * K
    tflops = {
        name: (flops / statistics.median(t) * 1e-9 if t else float("nan"))
        for name, t in times.items()
    }

    row = dict(
        M=M,
        N=N,
        K=K,
        bias=s["bias"],
        out=s["out"],
        tiles=s["tiles"],
        waves=round(s["tiles"] / CUS, 3),
        bucket=bucket(s["tiles"]),
        csv_pick=s["csv_pick"],
        csv_gluon=s["csv_gluon"],
        csv_aiter=s["csv_aiter"],
        csv_gap_pct=s["csv_gap"],
        live_libtype=libtype,
        live_kernel=kernel,
    )
    for name in fns:
        row[f"{name}_tflops"] = round(tflops[name], 1)
    for name in fns:
        row[f"{name}_ok"] = ok[name]
        row[f"{name}_maxerr"] = f"{errors[name]:.3g}"
    return row


def fmt(x, spec="{:.0f}"):
    return spec.format(x) if x == x else "-"


def summarize():
    rows = sorted(read_rows(), key=lambda r: (int(r["tiles"]), int(r["K"])))
    lines = [
        "| Shape | Tiles | Waves | Bucket | Live pick | Pick | v9 | v10 | v11 | v12 | v13 "
        "| Best Gluon | Live gap | CSV gap | Wrong |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    per_bucket = {}
    wins = {}
    for r in rows:
        tf = {v: float(r[f"{v}_tflops"]) for v in VARIANTS}
        gl = {v: tf[v] for v in GLUON if tf[v] == tf[v]}
        best = max(gl, key=gl.get) if gl else None
        gap = 100 * (tf["pick"] / gl[best] - 1) if best else float("nan")
        wins[best] = wins.get(best, 0) + 1
        per_bucket.setdefault(r["bucket"], []).append(
            (gap, float(r["csv_gap_pct"]), 100 * (gl.get("v9", float("nan")) / tf["pick"] - 1))
        )
        pick = r["live_libtype"] + (
            "" if r["live_libtype"] == r["csv_pick"] else f" (CSV: {r['csv_pick']})"
        )
        wrong = [v for v in VARIANTS if r[f"{v}_ok"] != "True"]
        shape = f"{r['M']}x{r['N']}x{r['K']}" + ("" if r["out"] == "bfloat16" else " fp32 out")
        lines.append(
            f"| {shape} | {r['tiles']} | {float(r['waves']):.2f} | {r['bucket']} | {pick} "
            f"| {fmt(tf['pick'])} | " + " | ".join(fmt(tf[v]) for v in GLUON)
            + f" | {best or '-'} | {fmt(gap, '{:+.1f}%')} | {float(r['csv_gap_pct']):+.1f}% "
            f"| {' '.join(wrong) or '-'} |"
        )
    lines += ["", "Live gap = pick / best Gluon - 1 (positive: AITER's pick is faster).", ""]
    lines += [
        "| Bucket | Shapes | Median live gap | Max live gap | Median CSV gap | Gluon (best) wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for b, xs in per_bucket.items():
        live = [g for g, _, _ in xs if g == g]
        lines.append(
            f"| {b} | {len(xs)} | {statistics.median(live):+.1f}% | {max(live):+.1f}% "
            f"| {statistics.median(c for _, c, _ in xs):+.1f}% | {sum(g <= 0 for g in live)} |"
        )
    lines += ["", "Best Gluon version per shape: "
              + ", ".join(f"{v}: {n}" for v, n in sorted(wins.items(), key=lambda t: -t[1]))]
    text = "\n".join(lines) + "\n"
    with open(SUMMARY, "w") as f:
        f.write(text)
    print(text)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--shapes", type=int, default=None, help="only the first N shapes")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--summarize", action="store_true")
    p.add_argument(
        "--tag", default="", help="suffix for the results files, e.g. hbl122 for another hipBLASLt"
    )
    args = p.parse_args()
    if args.tag:
        global OUT_CSV, SUMMARY
        OUT_CSV = os.path.join(RESULTS, f"aiter_live_compare_{args.tag}.csv")
        SUMMARY = os.path.join(RESULTS, f"aiter_live_compare_{args.tag}.md")
    if args.summarize:
        summarize()
        return
    os.makedirs(RESULTS, exist_ok=True)
    done = {(int(r["M"]), int(r["N"]), int(r["K"])) for r in read_rows()}
    shapes = load_shapes()[: args.shapes]
    for i, s in enumerate(shapes):
        if key(s) in done:
            continue
        print(f"[{i + 1}/{len(shapes)}] {s['M']}x{s['N']}x{s['K']} tiles={s['tiles']}", flush=True)
        row = run_shape(s, args.rounds)
        print(
            "    pick=%s %s | " % (row["live_libtype"], row["pick_tflops"])
            + " ".join(f"{v}={row[f'{v}_tflops']}" for v in GLUON)
            + " | wrong: "
            + (" ".join(v for v in VARIANTS if not row[f"{v}_ok"]) or "-"),
            flush=True,
        )
        append_row(row)
    summarize()


if __name__ == "__main__":
    main()
