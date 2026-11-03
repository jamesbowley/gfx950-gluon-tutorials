"""Tune one shape: every valid v31 tile spec and Stream-K method, every v9 tile, and
torch.mm (hipBLASLt), each timed over 1000 launches under rocprofv3.

    source kernels/gemm/intra_wave/a16w16/v31_streamk_aiter/env.sh; cd $REPO
    bash kernels/gemm/intra_wave/a16w16/v31_streamk_aiter/run_guarded.sh \
        experiments/tune_2561x6288x7168/results/tune.log python experiments/tune_2561x6288x7168/tune.py all

Points are ``KIND:BMxBNxBK:MxNxK`` as in streamk_v31_rule/grid.py (v31 kinds dp, ow{n},
rs{n}, l2rs{n}, 2t, sp), plus ``v9:BMxBNx64:...`` and ``torch:hbl:...``. Each point is
checked against an fp32 torch reference, run once over every rotating copy, then a HIP
graph of GRAPH_LEN launches over ROTATING_MB of rotating inputs (cold caches) is
replayed REPLAYS times: all TIMED traced dispatches go into the stats. Gluon points run
in shuffled chunks, one rocprofv3 process per chunk, each chunk's first point timed
once more and dropped; torch runs in its own process. hipBLASLt kernels per call (if
more than one) are summed per call.
"""

import argparse
import csv
import glob
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RULE = os.path.abspath(os.path.join(HERE, "..", "streamk_v31_rule"))
sys.path.insert(0, RULE)
import candidates  # noqa: E402
import gpu_guard  # noqa: E402
import grid  # noqa: E402  (puts kernels/gemm/intra_wave/a16w16 on sys.path)
import sweep  # noqa: E402

M, N, K = 2561, 6288, 7168
GRAPH_LEN = 100
REPLAYS = 10
TIMED = GRAPH_LEN * REPLAYS
ROTATING_MB = 2048
CHUNK = 40
SEED = 2561
SMI_INDEX = int(os.environ.get("V31_SMI_INDEX", "1"))
RESULTS = os.path.join(HERE, "results")
TMP = os.path.join(HERE, "tmp")
GLUON_KERNEL = "_gemm_a16w16_compute_bound_kernel"
HBL_KERNEL = "Cijk_"
V9_TILES = [f"{bm}x{bn}x64" for bm in (256, 128, 64) for bn in (256, 128, 64)]
FIELDS = ("point", "version", "tile", "method", "policy", "fixup", "split_n", "exchange", "T", "S",
          "correct", "us_median", "us_mean", "us_min", "us_p90", "tflops", "timed", "kernels_per_call",
          "hbl_tile", "hbl_kernel", "formula_pick", "error")


def v31_kinds(tile):
    """Every valid dp / owner / rs / l2rs split, two_tile and spread (no pdp: T > P)."""
    kinds = (["dp"] + [f"ow{n}" for n in grid.OWNER_N] + [f"rs{n}" for n in grid.RS_N]
             + [f"l2rs{n}" for n in grid.RS_N] + ["2t", "sp"])
    return [k for k in kinds if grid.valid(k, tile, M, N, K)]


def all_points():
    pts = [grid.point_name(k, t, M, N, K) for t in grid.ALL_SPECS for k in v31_kinds(t)]
    pts += [grid.point_name("v9", t, M, N, K) for t in V9_TILES]
    return pts + [grid.point_name("torch", "hbl", M, N, K)]


def formula_kinds():
    """The v31 default rule's plan at each spec, as a point kind."""
    rule = json.load(open(os.path.join(RULE, "results", "streamk_rule.json")))
    out = {}
    for spec in candidates.SPECS:
        c = candidates.streamk_default(M, N, K, spec, grid.P, rule)
        kind = "dp" if c["STREAMK_POLICY"] == "dp" else ("ow" if c["STREAMK_FIXUP"] == "owner" else "rs") + str(c["SPLIT_N"])
        out[candidates.spec_name(spec)] = kind
    return out


def version_of(kind):
    return kind if kind in ("v9", "torch") else "v31"


def method_of(kind):
    if kind == "torch":
        return "hipBLASLt"
    if kind in ("v9", "dp"):
        return "dp"
    if kind == "2t":
        return "two_tile"
    if kind == "sp":
        return "spread"
    for prefix, name in (("l2rs", "one_tile rs (l2)"), ("rs", "one_tile rs"), ("ow", "one_tile owner")):
        if kind.startswith(prefix):
            return f"{name} n={kind[len(prefix):]}"
    raise ValueError(kind)


def run_fn(kind, tile):
    if kind == "torch":
        import torch

        return lambda x, w, y: torch.mm(x, w.t(), out=y)
    if kind == "v9":
        from v31_streamk_aiter import dp_ab

        v9 = dp_ab.v9_module()
        bm, bn, _ = grid.parse_tile(tile)
        return lambda x, w, y: v9.gemm_a16w16_compute_bound(x, w, y, tile=(bm, bn))
    return sweep.run_fn(kind, tile)


def worker(args):
    import bench_ext as bench
    import torch

    dt = torch.bfloat16
    x = torch.rand((M, K), device="cuda", dtype=dt) - 0.5
    w = torch.rand((N, K), device="cuda", dtype=dt) - 0.5
    ref = (x.float() @ w.float().T).to(dt)
    a_l, b_l, c_l, copies = bench.gen_rotating_tensors(M, N, K, dt, ROTATING_MB)
    torch.cuda.synchronize()
    manifest = []
    for p in args.points.split(","):
        kind, tile, *_ = grid.parse_point(p)
        run = run_fn(kind, tile)
        y = torch.empty((M, N), device="cuda", dtype=dt)
        run(x, w, y)
        torch.cuda.synchronize()
        correct = bool(torch.allclose(y, ref, atol=1e-1, rtol=1e-2))
        for i in range(copies):
            run(a_l[i], b_l[i].T, c_l[i])
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(GRAPH_LEN):
                run(a_l[i % copies], b_l[i % copies].T, c_l[i % copies])
        for _ in range(REPLAYS):
            g.replay()
        torch.cuda.synchronize()
        del g
        manifest.append({"point": p, "calls": 1 + copies + TIMED, "correct": correct})
        print(f"{p} copies={copies} correct={correct}", flush=True)
    json.dump(manifest, open(args.manifest, "w"), indent=1)


def warm_worker(args):
    import torch
    from v31_streamk_aiter import dp_ab
    from v31_streamk_aiter import matmul_kernel as mk

    v9 = dp_ab.v9_module()
    mk._gemm_a16w16_compute_bound_kernel = sweep.Warmup(mk._gemm_a16w16_compute_bound_kernel)
    v9._gemm_a16w16_compute_bound_kernel = sweep.Warmup(v9._gemm_a16w16_compute_bound_kernel)
    x = torch.empty((M, K), device="cuda", dtype=torch.bfloat16)
    w = torch.empty((N, K), device="cuda", dtype=torch.bfloat16)
    for p in args.points.split(","):
        kind, tile, *_ = grid.parse_point(p)
        try:
            run_fn(kind, tile)(x, w, torch.empty((M, N), device="cuda", dtype=torch.bfloat16))
            print(f"WARM_OK {p}", flush=True)
        except Exception as e:  # noqa: BLE001  (reported; the point fails again when timed)
            print(f"WARM_FAIL {p} {type(e).__name__}: {e}", flush=True)


def warm(pts, jobs):
    todo = [p for p in pts if not p.startswith("torch:")]
    chunks = [todo[i::jobs] for i in range(jobs) if todo[i::jobs]]

    def one(c):
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "warm_worker", "--points", ",".join(c)],
                           capture_output=True, text=True)
        tail = [f"WARM_FAIL exit {r.returncode}: {r.stderr[-800:]}"] if r.returncode else []
        return r.stdout.splitlines() + tail

    with ThreadPoolExecutor(len(chunks)) as ex:
        lines = sum(ex.map(one, chunks), [])
    fails = [x for x in lines if x.startswith("WARM_FAIL")]
    for f in fails:
        print(f, flush=True)
    log(f"warm: {sum(x.startswith('WARM_OK') for x in lines)} of {len(todo)} compiled, {len(fails)} failures")


def per_call(rows, calls):
    """Per-call us of the last TIMED calls in rows (oldest first) whose final calls x k
    dispatches repeat with period k: k kernels per call."""
    for k in range(1, 5):
        if len(rows) < calls * k:
            break
        names = [r["Kernel_Name"] for r in rows[-calls * k:]]
        if all(names[i] == names[i % k] for i in range(len(names))):
            timed = rows[-TIMED * k:]
            us = [sum(int(r["End_Timestamp"]) - int(r["Start_Timestamp"]) for r in timed[i:i + k]) / 1e3
                  for i in range(0, len(timed), k)]
            return us, k, names[:k]
    raise RuntimeError(f"{len(rows)} dispatches for {calls} calls: no consistent kernels per call")


def timing(pts, out_dir):
    """Time pts in one rocprofv3 process: a result row per point."""
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    manifest = os.path.join(out_dir, "manifest.json")
    regex = HBL_KERNEL if pts[0].startswith("torch:") else GLUON_KERNEL
    cmd = ["rocprofv3", "--kernel-trace", "--kernel-include-regex", regex, "-d", out_dir, "-o", "run",
           "--output-format", "csv", "--", sys.executable, os.path.abspath(__file__), "worker",
           "--manifest", manifest, "--points", ",".join(pts)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if p.returncode or not os.path.exists(manifest):
        print(p.stdout[-3000:], p.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"rocprofv3 run failed ({p.returncode}) on {pts}")
    man = json.load(open(manifest))
    trace = glob.glob(os.path.join(out_dir, "**", "*kernel_trace.csv"), recursive=True)[0]
    rows = sorted((r for r in csv.DictReader(open(trace)) if r["Kernel_Name"].startswith(regex)),
                  key=lambda r: int(r["Dispatch_Id"]))
    out = []
    if regex == HBL_KERNEL:
        # One torch point per process; the fp32 reference's hipBLASLt kernels come first.
        assert len(man) == 1
        us, k, names = per_call(rows, man[0]["calls"])
        out.append(result_row(man[0]["point"], man[0]["correct"], us, k, names))
        return out
    expected = sum(m["calls"] for m in man)
    if len(rows) != expected:
        raise RuntimeError(f"{len(rows)} gluon dispatches in the trace, expected {expected}")
    pos = 0
    for m in man:
        us, k, names = per_call(rows[pos:pos + m["calls"]], m["calls"])
        pos += m["calls"]
        out.append(result_row(m["point"], m["correct"], us, k, names))
    return out


def hbl_tile(name):
    mt = re.search(r"_MT(\d+)x(\d+)x(\d+)_", name)
    sk = re.search(r"_(SK\d+)_", name)
    gsu = re.search(r"_GSU(\d+)_", name)
    s = f"{mt.group(1)}x{mt.group(2)}x{mt.group(3)}" if mt else "?"
    return s + (f" {sk.group(1)}" if sk else "") + (f" GSU{gsu.group(1)}" if gsu else "")


def base_row(p, formula):
    kind, tile, *_ = grid.parse_point(p)
    r = dict.fromkeys(FIELDS, "")
    r.update(point=p, version=version_of(kind), tile=tile, method=method_of(kind))
    if kind != "torch":
        sh = grid.shape_info(tile, M, N, K)
        r.update(T=sh["T"], S=sh["S"])
    if r["version"] == "v31":
        c = grid.config_for(kind, tile)
        r.update(policy=c["STREAMK_POLICY"], fixup=c["STREAMK_FIXUP"], split_n=c["SPLIT_N"] or 1,
                 exchange=c["STREAMK_EXCHANGE"], formula_pick=int(formula.get(tile) == kind))
    return r


FORMULA = {}


def result_row(p, correct, us, k, names):
    r = base_row(p, FORMULA)
    med = statistics.median(us)
    r.update(correct=correct, us_median=round(med, 3), us_mean=round(statistics.fmean(us), 3),
             us_min=round(min(us), 3), us_p90=round(sorted(us)[int(0.9 * len(us))], 3),
             tflops=round(2 * M * N * K / (med * 1e-6) / 1e12, 1), timed=len(us), kernels_per_call=k)
    if p.startswith("torch:"):
        r.update(hbl_tile=hbl_tile(names[0]), hbl_kernel=" | ".join(names))
    return r


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def check_gpu(when):
    if gpu_guard.foreign_seen(SMI_INDEX, os.getpid(), 3, 2.0):
        log(f"FOREIGN process on rocm-smi GPU {SMI_INDEX} {when}; stopping")
        sys.exit(4)


def time_all(pts, out):
    done = {r["point"] for r in csv.DictReader(open(out))} if os.path.exists(out) else set()
    gluon = [p for p in pts if not p.startswith("torch:") and p not in done]
    random.Random(SEED).shuffle(gluon)
    batches = [gluon[i:i + CHUNK] for i in range(0, len(gluon), CHUNK)]
    batches += [[p] for p in pts if p.startswith("torch:") and p not in done]
    out_dir = os.path.join(TMP, "trace")
    for i, b in enumerate(batches):
        check_gpu(f"before batch {i}")
        try:
            # A process's first timed point comes out a few percent fast: time it twice, drop the first.
            res = timing(b[:1] + b, out_dir)[1:] if not b[0].startswith("torch:") else timing(b, out_dir)
        except RuntimeError as e:
            log(f"batch {i} failed ({e}); timing its points one by one")
            res = []
            for p in b:
                try:
                    res += timing([p, p], out_dir)[1:] if not p.startswith("torch:") else timing([p], out_dir)
                except RuntimeError as e1:
                    res.append(dict(base_row(p, FORMULA), error=str(e1)[:200]))
        check_gpu(f"after batch {i} (its rows dropped)")
        new = not os.path.exists(out)
        with open(out, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                wr.writeheader()
            wr.writerows(res)
        log(f"batch {i + 1} / {len(batches)} done")


def report(out_csv, out_md):
    rows = [r for r in csv.DictReader(open(out_csv)) if r["us_median"]]
    bad = [r for r in csv.DictReader(open(out_csv)) if not r["us_median"] or r["correct"] != "True"]
    ok = sorted((r for r in rows if r["correct"] == "True"), key=lambda r: float(r["us_median"]))
    torch_r = next((r for r in ok if r["version"] == "torch"), None)
    t_us = float(torch_r["us_median"]) if torch_r else None

    def vs(r):
        return f"{t_us / float(r['us_median']):.3f}" if t_us else ""

    def tile_label(r):
        return f"{r['tile']} ({r['hbl_tile']})" if r["version"] == "torch" else r["tile"]

    L = [f"# {M}x{N}x{K} bf16: v31 Stream-K, v9 and hipBLASLt", "",
         f"{2 * M * N * K / 1e9:.1f} GFLOP per call. Each point: {TIMED} launches traced with rocprofv3 "
         f"(a graph of {GRAPH_LEN} replayed {REPLAYS} times) over {ROTATING_MB} MB of rotating inputs; "
         f"time is the median of all {TIMED}, TFLOPS from the median. `vs torch` is hipBLASLt's time over "
         "the point's (above 1 is faster than torch). `*` marks the v31 default formula's plan for that tile.",
         "", "## Summary", "", "| version | tile | method | median us | mean us | TFLOPS | vs torch |",
         "|---|---|---|---|---|---|---|"]
    for ver in ("v31", "v9", "torch"):
        best = next((r for r in ok if r["version"] == ver), None)
        if best:
            L.append(f"| {ver} (best) | {tile_label(best)} | {best['method']} | {best['us_median']} | "
                     f"{best['us_mean']} | {best['tflops']} | {vs(best)} |")
    L += ["", "## Best per tile", "",
          "| tile | best v31 method | v31 median us | v31 TFLOPS | v31 dp us | formula's plan (us) | v9 us | v9 TFLOPS |",
          "|---|---|---|---|---|---|---|---|"]
    by = {}
    for r in ok:
        by.setdefault((r["version"], r["tile"]), []).append(r)
    for spec in grid.ALL_SPECS:
        v31 = by.get(("v31", spec), [])
        if not v31:
            continue
        b = v31[0]
        dp = next((r for r in v31 if r["method"] == "dp"), None)
        fp = next((r for r in v31 if r["formula_pick"] == "1"), None)
        v9 = by.get(("v9", spec), [None])[0]
        L.append(f"| {spec} | {b['method']} | {b['us_median']} | {b['tflops']} | "
                 f"{dp['us_median'] if dp else ''} | {(fp['method'] + ' (' + fp['us_median'] + ')') if fp else ''} | "
                 f"{v9['us_median'] if v9 else ''} | {v9['tflops'] if v9 else ''} |")
    L += ["", "## Every point, fastest first", "",
          "| # | version | tile | method | median us | mean us | min us | p90 us | TFLOPS | vs torch |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(ok, 1):
        mark = " *" if r["formula_pick"] == "1" else ""
        L.append(f"| {i} | {r['version']} | {tile_label(r)} | {r['method']}{mark} | {r['us_median']} | "
                 f"{r['us_mean']} | {r['us_min']} | {r['us_p90']} | {r['tflops']} | {vs(r)} |")
    if torch_r:
        L += ["", f"hipBLASLt kernel(s) per call ({torch_r['kernels_per_call']}): `{torch_r['hbl_kernel']}`"]
    if bad:
        L += ["", "## Failed or incorrect", ""] + [f"- {r['point']}: correct={r['correct']} {r['error']}" for r in bad]
    open(out_md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:20 + len(grid.ALL_SPECS)]))


def main():
    global FORMULA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["all", "list", "warm", "timing", "report", "worker", "warm_worker"])
    ap.add_argument("--points")
    ap.add_argument("--manifest")
    ap.add_argument("--jobs", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(RESULTS, "tune.csv"))
    args = ap.parse_args()
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(TMP, "triton"))
    if args.mode == "worker":
        return worker(args)
    if args.mode == "warm_worker":
        return warm_worker(args)
    FORMULA = formula_kinds()
    pts = all_points()
    os.makedirs(RESULTS, exist_ok=True)
    if args.mode == "list":
        print("\n".join(pts))
        print(f"{len(pts)} points; formula plan per spec: {FORMULA}", file=sys.stderr)
        return
    if args.mode in ("all", "warm"):
        warm(pts, args.jobs)
    if args.mode in ("all", "timing"):
        time_all(pts, args.out)
    if args.mode in ("all", "report"):
        report(args.out, os.path.splitext(args.out)[0] + ".md")


if __name__ == "__main__":
    main()
